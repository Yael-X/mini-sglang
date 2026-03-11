"""Tests for FP8 input quantization kernels."""

from __future__ import annotations

import pytest
import torch

from minisgl.kernel.input_quant import (
    FP8InputQuantizer,
    FP8_E4M3_MAX,
    compute_fp8_scale,
    dequantize_fp8,
    prepare_fp8_gemm_input,
    quantize_input_to_fp8,
    quantize_to_fp8,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestComputeFP8Scale:
    """Tests for compute_fp8_scale function."""

    def test_per_tensor_scale_shape(self) -> None:
        """Test per-tensor scale returns scalar."""
        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.bfloat16, device="cuda")
        scale = compute_fp8_scale(x, scale_method="per_tensor")
        assert scale.numel() == 1
        assert scale.item() > 0

    def test_per_token_scale_shape(self) -> None:
        """Test per-token scale returns [M, 1] shape."""
        x = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.bfloat16, device="cuda")
        scale = compute_fp8_scale(x, scale_method="per_token")
        assert scale.shape == (2, 1)
        assert (scale > 0).all()

    def test_per_token_scale_3d(self) -> None:
        """Test per-token scale with 3D input."""
        x = torch.randn(2, 4, 8, dtype=torch.bfloat16, device="cuda")
        scale = compute_fp8_scale(x, scale_method="per_token")
        assert scale.shape == (2 * 4, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestQuantizeInputToFP8:
    """Tests for quantize_input_to_fp8 function."""

    def test_basic_quantization(self) -> None:
        """Test basic quantization roundtrip."""
        x = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")

        assert x_fp8.dtype == torch.float8_e4m3fn
        assert x_fp8.shape == x.shape
        assert x_scale.numel() == 1

    def test_per_token_quantization(self) -> None:
        """Test per-token quantization."""
        x = torch.randn(2, 16, 32, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_token")

        assert x_fp8.dtype == torch.float8_e4m3fn
        assert x_fp8.shape == x.shape
        assert x_scale.shape == (32, 1)  # batch * seq = 2 * 16 = 32

    def test_quantization_error(self) -> None:
        """Test quantization error is within acceptable bounds."""
        torch.manual_seed(42)
        x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_token")
        x_dequant = dequantize_fp8(x_fp8, x_scale)

        # FP8 has ~3-4 bits of mantissa, expect ~3-5% relative error
        relative_error = (x_dequant - x).abs().max() / x.abs().max()
        assert relative_error < 0.05, f"Relative error {relative_error:.4f} exceeds 5%"

    def test_small_values(self) -> None:
        """Test quantization of small values."""
        x = torch.full((16, 16), 0.001, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")
        x_dequant = dequantize_fp8(x_fp8, x_scale)

        # Small values should be preserved reasonably
        relative_error = (x_dequant - x).abs().mean() / x.abs().mean()
        assert relative_error < 0.1

    def test_large_values(self) -> None:
        """Test quantization of large values near FP8 max."""
        x = torch.full((16, 16), 400.0, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")
        x_dequant = dequantize_fp8(x_fp8, x_scale)

        # Large values should be preserved
        assert x_dequant.abs().max() <= FP8_E4M3_MAX * 1.01  # Allow small margin

    def test_non_cuda_raises(self) -> None:
        """Test that non-CUDA input raises error."""
        x = torch.randn(16, 16, dtype=torch.bfloat16, device="cpu")

        with pytest.raises(ValueError, match="must be on CUDA"):
            quantize_input_to_fp8(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestDequantizeFP8:
    """Tests for dequantize_fp8 function."""

    def test_roundtrip(self) -> None:
        """Test quantize -> dequantize roundtrip."""
        x = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_tensor")
        x_back = dequantize_fp8(x_fp8, x_scale, dtype=torch.bfloat16)

        assert x_back.dtype == torch.bfloat16
        assert x_back.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFP8InputQuantizer:
    """Tests for FP8InputQuantizer class."""

    def test_callable(self) -> None:
        """Test that quantizer is callable."""
        quantizer = FP8InputQuantizer(scale_method="per_tensor")
        x = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = quantizer(x)

        assert x_fp8.dtype == torch.float8_e4m3fn
        assert x_fp8.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestPrepareFP8GEMMInput:
    """Tests for prepare_fp8_gemm_input function."""

    def test_2d_input(self) -> None:
        """Test with 2D input."""
        x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = prepare_fp8_gemm_input(x, scale_method="per_token")

        assert x_fp8.shape == (64, 128)
        assert x_fp8.is_contiguous()

    def test_3d_input_reshaped(self) -> None:
        """Test that 3D input is reshaped to 2D."""
        x = torch.randn(2, 32, 128, dtype=torch.bfloat16, device="cuda")

        x_fp8, x_scale = prepare_fp8_gemm_input(x, scale_method="per_token")

        # Should be reshaped to [64, 128]
        assert x_fp8.shape == (64, 128)
        assert x_fp8.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestQuantizationPrecision:
    """Tests for quantization precision."""

    def test_per_token_better_than_per_tensor(self) -> None:
        """Test that per-token quantization has lower error for varied data."""
        torch.manual_seed(42)

        # Create data with varied magnitudes
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        x[:, :128] *= 100  # First half large values
        x[:, 128:] *= 0.01  # Second half small values

        # Per-tensor
        x_fp8_pt, x_scale_pt = quantize_input_to_fp8(x, scale_method="per_tensor")
        x_dequant_pt = dequantize_fp8(x_fp8_pt, x_scale_pt)
        error_pt = (x_dequant_pt - x).abs().mean()

        # Per-token
        x_fp8_pkt, x_scale_pkt = quantize_input_to_fp8(x, scale_method="per_token")
        x_dequant_pkt = dequantize_fp8(x_fp8_pkt, x_scale_pkt)
        error_pkt = (x_dequant_pkt - x).abs().mean()

        # Per-token should have lower error for data with varied magnitudes
        assert error_pkt < error_pt, "Per-token should be more precise for varied data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])