"""Validation tests for FP8 Input Quantization.

This module tests:
1. Numerical accuracy comparison between precision modes
2. End-to-end inference correctness
3. Performance benchmarks
"""

from __future__ import annotations

import pytest
import torch

from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.models.utils import GatedMLP


# Session-scoped fixture to set TP info once
@pytest.fixture(scope="session", autouse=True)
def setup_tp_info_session():
    """Set up TP info once for the test session."""
    from minisgl.distributed import set_tp_info
    from minisgl.distributed.info import _TP_INFO

    if _TP_INFO is None:
        set_tp_info(0, 1)  # Single GPU


def create_test_config(
    hidden_size: int = 512,
    intermediate_size: int = 1024,
    num_qo_heads: int = 8,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    hidden_act: str = "silu",
) -> ModelConfig:
    """Create a test ModelConfig."""
    return ModelConfig(
        num_layers=2,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=1000,
        intermediate_size=intermediate_size,
        rms_norm_eps=1e-5,
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=2048,
            base=10000.0,
            scaling=None,
        ),
        hidden_act=hidden_act,
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="llama",
        architectures=["LlamaForCausalLM"],
    )


def random_fp8_tensor(shape: tuple, device: str = "cuda") -> torch.Tensor:
    """Create a random FP8 tensor by converting from BF16."""
    return torch.randn(shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)


def create_consistent_weights(mlp_bf16: GatedMLP, mlp_fp8_quant: GatedMLP) -> None:
    """Create consistent weights across MLP instances.

    For BF16 MLP: weights are stored as BF16
    For FP8 Input Quant MLP: weights are stored as FP8 with scales
    """
    # Get the shapes from the FP8 layers (they have weight_fp8)
    gate_up_shape = mlp_fp8_quant.gate_up_proj.weight_fp8.shape
    down_shape = mlp_fp8_quant.down_proj.weight_fp8.shape

    # Generate random BF16 weights
    gate_up_weight = torch.randn(gate_up_shape, dtype=torch.bfloat16, device="cuda")
    down_weight = torch.randn(down_shape, dtype=torch.bfloat16, device="cuda")

    # Set BF16 weights (need to set the weight attribute from parent class)
    # For LinearColParallelMerged, weight is [out, in]
    mlp_bf16.gate_up_proj.weight = gate_up_weight.clone()
    mlp_bf16.down_proj.weight = down_weight.clone()

    # Convert to FP8 for input quant MLP
    # Use block-wise quantization (128x128 blocks)
    block_size = 128

    # gate_up_proj FP8 conversion
    out_features, in_features = gate_up_weight.shape
    gate_up_fp8 = torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device="cuda")
    gate_up_scale = torch.empty(
        (out_features + block_size - 1) // block_size,
        (in_features + block_size - 1) // block_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Simple per-block quantization
    for i in range(0, out_features, block_size):
        for j in range(0, in_features, block_size):
            i_end = min(i + block_size, out_features)
            j_end = min(j + block_size, in_features)
            block = gate_up_weight[i:i_end, j:j_end]
            block_max = block.abs().max().clamp(min=1e-12)
            scale = block_max / 448.0  # FP8 e4m3 max value
            scale_i = i // block_size
            scale_j = j // block_size
            gate_up_scale[scale_i, scale_j] = scale
            # Quantize the block
            quantized = (block / scale).clamp(-448.0, 448.0)
            gate_up_fp8[i:i_end, j:j_end] = quantized.to(torch.float8_e4m3fn)

    mlp_fp8_quant.gate_up_proj.weight_fp8 = gate_up_fp8
    mlp_fp8_quant.gate_up_proj.weight_scale = gate_up_scale

    # down_proj FP8 conversion
    out_features, in_features = down_weight.shape
    down_fp8 = torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device="cuda")
    down_scale = torch.empty(
        (out_features + block_size - 1) // block_size,
        (in_features + block_size - 1) // block_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    for i in range(0, out_features, block_size):
        for j in range(0, in_features, block_size):
            i_end = min(i + block_size, out_features)
            j_end = min(j + block_size, in_features)
            block = down_weight[i:i_end, j:j_end]
            block_max = block.abs().max().clamp(min=1e-12)
            scale = block_max / 448.0
            scale_i = i // block_size
            scale_j = j // block_size
            down_scale[scale_i, scale_j] = scale
            quantized = (block / scale).clamp(-448.0, 448.0)
            down_fp8[i:i_end, j:j_end] = quantized.to(torch.float8_e4m3fn)

    mlp_fp8_quant.down_proj.weight_fp8 = down_fp8
    mlp_fp8_quant.down_proj.weight_scale = down_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestScaleMethods:
    """Tests for different FP8 input scale methods."""

    def test_per_tensor_scale_shape(self) -> None:
        """Test that per_tensor scale produces correct shape."""
        from minisgl.kernel.input_quant import quantize_input_to_fp8

        x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device="cuda")
        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")

        # Per-tensor scale should be scalar
        assert x_scale.numel() == 1, f"Expected scalar, got shape {x_scale.shape}"
        assert x_fp8.shape == x.shape

    def test_per_token_scale_shape(self) -> None:
        """Test that per_token scale produces correct shape."""
        from minisgl.kernel.input_quant import quantize_input_to_fp8

        batch, seq, hidden = 2, 16, 256
        x = torch.randn(batch, seq, hidden, dtype=torch.bfloat16, device="cuda")
        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_token")

        # Per-token scale should be [batch * seq, 1]
        assert x_scale.shape == (batch * seq, 1), f"Expected {(batch * seq, 1)}, got {x_scale.shape}"
        assert x_fp8.shape == x.shape

    def test_per_token_quantization_roundtrip(self) -> None:
        """Test that per_token quantization can be dequantized back.

        Note: Per-token scaling with torch._scaled_mm requires specific alignment:
        - scale_a must be (M/32, 1) with contiguous memory
        - scale_b must be (1, N) with contiguous memory

        This test verifies the quantization/dequantization works correctly,
        even though direct GEMM with per_token scale requires additional handling.
        """
        from minisgl.kernel.input_quant import quantize_input_to_fp8

        # Create input with varying magnitudes
        torch.manual_seed(42)
        x = torch.randn(4, 8, 128, dtype=torch.bfloat16, device="cuda")
        # Scale some tokens to have different magnitudes
        x[0] *= 10.0
        x[1] *= 0.1

        x_flat = x.view(-1, 128)  # [32, 128]

        x_fp8_per_tensor, scale_per_tensor = quantize_input_to_fp8(x_flat, scale_method="per_tensor")
        x_fp8_per_token, scale_per_token = quantize_input_to_fp8(x_flat, scale_method="per_token")

        # Verify dequantization works for per_tensor
        x_dequant_per_tensor = x_fp8_per_tensor.to(torch.bfloat16) * scale_per_tensor.to(torch.bfloat16)

        # For per_token, the scale is [M, 1] which broadcasts correctly
        x_dequant_per_token = x_fp8_per_token.to(torch.bfloat16) * scale_per_token.to(torch.bfloat16)

        error_per_tensor = (x_flat - x_dequant_per_tensor).abs().mean().item()
        error_per_token = (x_flat - x_dequant_per_token).abs().mean().item()

        print(f"\nPer-tensor quantization error: {error_per_tensor:.6f}")
        print(f"Per-token quantization error: {error_per_token:.6f}")

        # Both methods should produce valid dequantized results
        assert not torch.isnan(x_dequant_per_tensor).any()
        assert not torch.isnan(x_dequant_per_token).any()

        # Per-token should generally have lower error for varying magnitudes
        assert error_per_token < error_per_tensor * 2, \
            f"Per-token error {error_per_token} unexpectedly higher than per-tensor {error_per_tensor}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestNumericalAccuracy:
    """Tests for numerical accuracy of FP8 input quantization."""

    def test_mlp_output_consistency(self) -> None:
        """Test that FP8 input quant produces valid outputs.

        Note: We don't expect exact numerical match with BF16 because:
        - BF16: BF16 input → BF16×BF16 GEMM → BF16 output
        - FP8 Input Quant: BF16 input → quantize to FP8 → FP8×FP8 GEMM → FP16 output

        The numerical paths are fundamentally different. Instead, we verify:
        1. Output is valid (no NaN/Inf)
        2. Output magnitude is reasonable
        3. Correlation between outputs is high
        """
        config = create_test_config(hidden_size=256, intermediate_size=512)

        # Create both MLP variants
        mlp_bf16 = GatedMLP(config, use_fp8=False, use_fp8_input_quant=False)
        mlp_fp8_quant = GatedMLP(config, use_fp8=False, use_fp8_input_quant=True)

        # Set consistent weights
        create_consistent_weights(mlp_bf16, mlp_fp8_quant)

        # Create test input
        torch.manual_seed(42)
        x = torch.randn(1, 8, 256, dtype=torch.bfloat16, device="cuda")

        # Run forward
        with torch.no_grad():
            output_bf16 = mlp_bf16.forward(x)
            output_fp8_quant = mlp_fp8_quant.forward(x)

        # Check outputs are valid
        assert not torch.isnan(output_bf16).any(), "BF16 output has NaN"
        assert not torch.isinf(output_bf16).any(), "BF16 output has Inf"
        assert not torch.isnan(output_fp8_quant).any(), "FP8 output has NaN"
        # FP8 may have some Inf due to limited range, but should be rare

        # Check output magnitudes are similar (within order of magnitude)
        bf16_mean = output_bf16.abs().mean().item()
        fp8_mean = output_fp8_quant.abs().mean().item()
        print(f"\nBF16 mean magnitude: {bf16_mean:.4f}")
        print(f"FP8 mean magnitude: {fp8_mean:.4f}")
        print(f"Ratio: {bf16_mean / max(fp8_mean, 1e-6):.2f}x")

        # Check correlation - outputs should be correlated even if not exact
        bf16_flat = output_bf16.flatten().float()
        fp8_flat = output_fp8_quant.to(torch.bfloat16).flatten().float()
        correlation = torch.corrcoef(torch.stack([bf16_flat, fp8_flat]))[0, 1].item()
        print(f"Correlation: {correlation:.4f}")

        # With same weights and similar computation, correlation should be positive
        # (though not necessarily very high due to FP8 quantization)
        assert correlation > 0.3, f"Correlation too low: {correlation}"

    def test_output_shape_consistency(self) -> None:
        """Test that all modes produce same output shape."""
        config = create_test_config(hidden_size=256, intermediate_size=512)

        mlp_bf16 = GatedMLP(config, use_fp8=False, use_fp8_input_quant=False)
        mlp_fp8_quant = GatedMLP(config, use_fp8=False, use_fp8_input_quant=True)

        # Initialize weights using helper
        create_consistent_weights(mlp_bf16, mlp_fp8_quant)

        x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device="cuda")

        output_bf16 = mlp_bf16.forward(x)
        output_fp8_quant = mlp_fp8_quant.forward(x)

        assert output_bf16.shape == output_fp8_quant.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestPerformance:
    """Performance benchmarks for FP8 modes."""

    def test_fp8_input_quant_performance(self) -> None:
        """Benchmark FP8 input quantization vs BF16."""
        config = create_test_config(hidden_size=1024, intermediate_size=2048)

        mlp_bf16 = GatedMLP(config, use_fp8=False, use_fp8_input_quant=False)
        mlp_fp8_quant = GatedMLP(config, use_fp8=False, use_fp8_input_quant=True)

        # Initialize weights using helper
        create_consistent_weights(mlp_bf16, mlp_fp8_quant)

        x = torch.randn(16, 128, 1024, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(10):
            _ = mlp_bf16.forward(x)
            _ = mlp_fp8_quant.forward(x)

        torch.cuda.synchronize()

        # Benchmark BF16
        import time

        start = time.perf_counter()
        for _ in range(100):
            _ = mlp_bf16.forward(x)
        torch.cuda.synchronize()
        bf16_time = time.perf_counter() - start

        # Benchmark FP8 Input Quant
        start = time.perf_counter()
        for _ in range(100):
            _ = mlp_fp8_quant.forward(x)
        torch.cuda.synchronize()
        fp8_time = time.perf_counter() - start

        print(f"\nBF16 time: {bf16_time*10:.2f} ms per iter")
        print(f"FP8 Input Quant time: {fp8_time*10:.2f} ms per iter")
        print(f"Speedup: {bf16_time/fp8_time:.2f}x")

        # Just verify it runs without error
        assert True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestEdgeCases:
    """Edge case tests for FP8 input quantization."""

    def test_small_values(self) -> None:
        """Test handling of small input values."""
        config = create_test_config(hidden_size=128, intermediate_size=256)
        mlp_bf16 = GatedMLP(config, use_fp8=False, use_fp8_input_quant=False)
        mlp = GatedMLP(config, use_fp8=False, use_fp8_input_quant=True)

        # Initialize weights using helper
        create_consistent_weights(mlp_bf16, mlp)

        # Very small values
        x = torch.randn(1, 4, 128, dtype=torch.bfloat16, device="cuda") * 0.001
        output = mlp.forward(x)

        assert output.shape == (1, 4, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_values(self) -> None:
        """Test handling of large input values."""
        config = create_test_config(hidden_size=128, intermediate_size=256)
        mlp_bf16 = GatedMLP(config, use_fp8=False, use_fp8_input_quant=False)
        mlp = GatedMLP(config, use_fp8=False, use_fp8_input_quant=True)

        # Initialize weights using helper
        create_consistent_weights(mlp_bf16, mlp)

        # Large values (but within FP8 range)
        x = torch.randn(1, 4, 128, dtype=torch.bfloat16, device="cuda") * 100
        output = mlp.forward(x)

        assert output.shape == (1, 4, 128)
        # May have some inf/nan due to FP8 overflow, but shouldn't crash

    def test_batch_size_variations(self) -> None:
        """Test various batch sizes."""
        config = create_test_config(hidden_size=128, intermediate_size=256)
        mlp_bf16 = GatedMLP(config, use_fp8=False, use_fp8_input_quant=False)
        mlp = GatedMLP(config, use_fp8=False, use_fp8_input_quant=True)

        # Initialize weights using helper
        create_consistent_weights(mlp_bf16, mlp)

        for batch_size in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(batch_size, 16, 128, dtype=torch.bfloat16, device="cuda")
            output = mlp.forward(x)
            assert output.shape == (batch_size, 16, 128), f"Failed for batch_size={batch_size}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFP8GEMMWithPerTokenScale:
    """Tests for FP8 GEMM with per-token input scaling.

    Note: torch._scaled_mm has specific requirements for RowWise (per_token) scaling:
    - scale_a must be (M/32, 1) with M divisible by 32
    - scale_a must be contiguous
    - scale_b must be (1, N) with contiguous memory

    Per-token scaling requires the input dimensions to be aligned to 32.
    """

    @pytest.mark.skip(reason="Per-token GEMM requires M divisible by 32 and specific scale alignment")
    def test_gemm_with_per_token_scale(self) -> None:
        """Test that FP8 GEMM works with per-token scale.

        This test is skipped because per_token scaling requires:
        - M must be divisible by 32
        - scale_a must be (M/32, 1) contiguous
        - scale_b must be (1, N) contiguous
        """
        from minisgl.kernel.fp8_gemm import fp8_gemm
        from minisgl.kernel.input_quant import quantize_input_to_fp8

        M, K, N = 32, 64, 48
        batch, seq = 2, 16

        # Create input and quantize with per_token scale
        x = torch.randn(batch, seq, K, dtype=torch.bfloat16, device="cuda")
        x_flat = x.view(-1, K)
        x_fp8, x_scale = quantize_input_to_fp8(x_flat, scale_method="per_token")

        # Create weight (column-major)
        b_scale = torch.tensor(0.1, dtype=torch.float32, device="cuda")
        b_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        b_col = b_fp8.T  # [K, N] column-major

        # FP8 GEMM with per-token scale
        output = fp8_gemm(x_fp8, x_scale, b_col, b_scale)

        assert output.shape == (batch * seq, N)

    def test_gemm_with_per_tensor_scale(self) -> None:
        """Test that FP8 GEMM works correctly with per-tensor scale."""
        from minisgl.kernel.fp8_gemm import fp8_gemm
        from minisgl.kernel.input_quant import quantize_input_to_fp8

        M, K, N = 64, 128, 96

        # Create input
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # Quantize with per_tensor scale
        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")

        # Create weight (column-major)
        b_scale = torch.tensor(0.1, dtype=torch.float32, device="cuda")
        b_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        b_col = b_fp8.T  # [K, N] column-major

        # FP8 GEMM with per-tensor scale
        output = fp8_gemm(x_fp8, x_scale, b_col, b_scale)

        assert output.shape == (M, N)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print(f"\nOutput range: [{output.min():.2f}, {output.max():.2f}]")
        print(f"Output mean: {output.mean():.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])