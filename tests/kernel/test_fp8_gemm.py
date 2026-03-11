"""Tests for FP8 GEMM operations."""

from __future__ import annotations

import pytest
import torch

from minisgl.kernel.fp8_gemm import (
    FP8GEMM,
    convert_bf16_weight_to_fp8,
    fp8_gemm,
    fp8_gemm_with_block_scale,
    prepare_weight_for_fp8_gemm,
)
from minisgl.kernel.input_quant import quantize_input_to_fp8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFP8GEMM:
    """Tests for fp8_gemm function."""

    def test_basic_gemm(self) -> None:
        """Test basic FP8 GEMM operation."""
        # Dimensions must be divisible by 16 for _scaled_mm
        M, K, N = 64, 32, 48

        # Create inputs
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

        # Quantize A (row-major)
        a_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        a_fp8 = (a_bf16 / a_scale).to(torch.float8_e4m3fn)

        # Quantize B and prepare as column-major
        b_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        b_NK = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / b_scale).to(torch.float8_e4m3fn)
        b_col_KN = b_NK.T  # Column-major view

        # Run FP8 GEMM
        result = fp8_gemm(a_fp8, a_scale, b_col_KN, b_scale)

        assert result.shape == (M, N)
        assert result.dtype == torch.float16

    def test_gemm_with_bias(self) -> None:
        """Test FP8 GEMM with bias."""
        # Use dimensions divisible by 16
        M, K, N = 32, 32, 32

        a_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        a_fp8 = (a_bf16 / a_scale).to(torch.float8_e4m3fn)

        b_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        b_NK = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        b_fp8_NK = (b_NK / b_scale).to(torch.float8_e4m3fn)
        b_col = b_fp8_NK.T  # Column-major [K, N]

        bias = torch.randn(N, dtype=torch.float16, device="cuda")

        result = fp8_gemm(a_fp8, a_scale, b_col, b_scale, bias=bias)

        assert result.shape == (M, N)

    def test_gemm_precision(self) -> None:
        """Test FP8 GEMM precision against BF16 reference."""
        torch.manual_seed(42)
        # Use dimensions divisible by 16
        M, K, N = 128, 64, 96

        # Create BF16 inputs
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        # Quantize with proper scales
        a_max = a_bf16.abs().max()
        a_scale = torch.tensor((a_max / 448.0).item(), dtype=torch.float32, device="cuda")
        a_fp8 = (a_bf16 / a_scale).to(torch.float8_e4m3fn)

        b_max = b_bf16.abs().max()
        b_scale = torch.tensor((b_max / 448.0).item(), dtype=torch.float32, device="cuda")
        b_fp8_NK = (b_bf16 / b_scale).to(torch.float8_e4m3fn)
        b_col = b_fp8_NK.T  # Column-major [K, N]

        # FP8 GEMM
        result_fp8 = fp8_gemm(a_fp8, a_scale, b_col, b_scale)

        # BF16 reference - note b_bf16 is [N, K], so we need to transpose for matmul
        result_bf16 = torch.matmul(a_bf16.float(), b_bf16.T.float())

        # Check relative error (FP8 precision ~3-5%)
        relative_error = (result_fp8.float() - result_bf16).abs().max() / result_bf16.abs().max()
        assert relative_error < 0.1, f"Relative error {relative_error:.4f} exceeds 10%"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFP8GEMMClass:
    """Tests for FP8GEMM class."""

    def test_callable(self) -> None:
        """Test FP8GEMM as callable."""
        M, K, N = 32, 32, 32

        # Create weight in column-major [K, N]
        b_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        b_NK = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        b_fp8_NK = (b_NK / b_scale).to(torch.float8_e4m3fn)
        b_col = b_fp8_NK.T  # Column-major [K, N]

        gemm = FP8GEMM(b_col, b_scale)

        a_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        a_fp8 = (a_bf16 / a_scale).to(torch.float8_e4m3fn)

        result = gemm(a_fp8, a_scale)

        assert result.shape == (M, N)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestConvertBF16WeightToFP8:
    """Tests for convert_bf16_weight_to_fp8 function."""

    def test_conversion_shape(self) -> None:
        """Test weight conversion preserves shape."""
        weight = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")

        weight_fp8, weight_scale = convert_bf16_weight_to_fp8(weight)

        assert weight_fp8.shape == weight.shape
        assert weight_fp8.dtype == torch.float8_e4m3fn
        assert weight_scale.dtype == torch.bfloat16

    def test_block_scale_shape(self) -> None:
        """Test block-wise scale has correct shape."""
        block_size = 128
        N, K = 256, 512

        weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        weight_fp8, weight_scale = convert_bf16_weight_to_fp8(weight, block_size=block_size)

        expected_scale_shape = ((N + block_size - 1) // block_size, (K + block_size - 1) // block_size)
        assert weight_scale.shape == expected_scale_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFP8GEMMWithBlockScale:
    """Tests for fp8_gemm_with_block_scale function."""

    def test_basic(self) -> None:
        """Test FP8 GEMM with block-wise scale (using max scale)."""
        # Use dimensions divisible by 16 and aligned with block size
        M, K, N = 64, 128, 96
        block_size = 128

        # Create activation - use per-tensor for TensorWise scaling
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        a_fp8, a_scale = quantize_input_to_fp8(a_bf16, scale_method="per_tensor")

        # Create weight with block-wise scale
        b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        b_fp8, b_scale = convert_bf16_weight_to_fp8(b_bf16, block_size=block_size)

        # Note: b_fp8 is [N, K], transpose to get column-major [K, N]
        result = fp8_gemm_with_block_scale(
            a_fp8, a_scale, b_fp8.T, b_scale, block_size=block_size
        )

        assert result.shape == (M, N)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFP8GEMMEndToEnd:
    """End-to-end tests for FP8 GEMM pipeline."""

    def test_linear_layer_simulation(self) -> None:
        """Simulate a linear layer with FP8 GEMM."""
        torch.manual_seed(42)

        # Use dimensions divisible by 16
        batch, seq, hidden, out_features = 2, 16, 64, 128
        M = batch * seq
        K = hidden
        N = out_features

        # Input
        x = torch.randn(batch, seq, hidden, dtype=torch.bfloat16, device="cuda")

        # Weight (simulating pre-quantized FP8 weight)
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        weight_fp8, weight_scale = convert_bf16_weight_to_fp8(weight_bf16)

        # Quantize input - use per-tensor for TensorWise scaling
        x_flat = x.view(M, K)
        x_fp8, x_scale = quantize_input_to_fp8(x_flat, scale_method="per_tensor")

        # FP8 GEMM
        # weight_fp8 is [N, K], transpose to [K, N] column-major
        weight_col = weight_fp8.T

        # Convert scale to float32 for _scaled_mm
        b_scale_float = weight_scale.max().float()
        output = fp8_gemm(x_fp8, x_scale, weight_col, b_scale_float)

        assert output.shape == (M, N)

        # Reshape back
        output = output.view(batch, seq, N)
        assert output.shape == (batch, seq, out_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])