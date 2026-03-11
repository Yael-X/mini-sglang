"""FP8 input/activation quantization kernels.

This module provides functions to quantize BF16 activations to FP8 format
for use with FP8 Tensor Core GEMM operations.
"""

from __future__ import annotations

from typing import Tuple

import torch

# FP8 e4m3fn maximum value
FP8_E4M3_MAX = 448.0


def quantize_to_fp8(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 using a pre-computed or computed scale.

    Args:
        x: Input tensor in BF16 or FP16
        scale: Optional pre-computed scale. If None, computed from tensor max.

    Returns:
        Tuple of (quantized FP8 tensor, scale tensor)
    """
    if scale is None:
        scale = compute_fp8_scale(x)

    x_fp8 = (x / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


def compute_fp8_scale(
    x: torch.Tensor,
    scale_method: str = "per_tensor",
) -> torch.Tensor:
    """Compute FP8 quantization scale.

    Args:
        x: Input tensor
        scale_method: "per_tensor" or "per_token"

    Returns:
        Scale tensor
    """
    if scale_method == "per_token":
        # Per-token: flatten to 2D first, then compute max along last dimension
        # Shape: [batch, seq, hidden] -> [batch * seq, hidden] -> [batch * seq, 1]
        x_flat = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
        max_val = x_flat.abs().amax(dim=-1, keepdim=True)
        scale = max_val / FP8_E4M3_MAX
    elif scale_method == "per_tensor":
        # Per-tensor: single scale for entire tensor
        max_val = x.abs().amax()
        scale = max_val / FP8_E4M3_MAX
    else:
        raise ValueError(f"Unknown scale_method: {scale_method}")

    # Clamp to avoid division by zero
    scale = scale.clamp(min=1e-12)

    # Return in float32 for _scaled_mm compatibility
    return scale.float()


def quantize_input_to_fp8(
    x: torch.Tensor,
    scale_method: str = "per_token",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input activations to FP8 format.

    This is the main entry point for input quantization in the FP8 inference
    pipeline. It quantizes BF16/FP16 activations to FP8 e4m3fn format.

    Args:
        x: Input tensor [batch, seq, hidden] or [M, K] in BF16 or FP16
        scale_method: Scale computation method
            - "per_token": Compute scale per token (recommended for better precision)
            - "per_tensor": Use single scale for entire tensor (faster but less precise)

    Returns:
        Tuple of:
            - x_fp8: FP8 quantized tensor with same shape as input
            - x_scale: Scale tensor
                - per_token: [M, 1] where M = batch * seq
                - per_tensor: scalar tensor

    Example:
        >>> x = torch.randn(2, 128, 4096, dtype=torch.bfloat16, device='cuda')
        >>> x_fp8, x_scale = quantize_input_to_fp8(x, scale_method="per_token")
        >>> # Use with FP8 GEMM
        >>> out = fp8_gemm(x_fp8, x_scale, weight_fp8, weight_scale)
    """
    # Ensure input is on CUDA
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    original_shape = x.shape

    # Compute scale
    x_scale = compute_fp8_scale(x, scale_method=scale_method)

    # For per-token with 3D input, we need to flatten, quantize, then reshape
    if scale_method == "per_token" and x.dim() > 2:
        x_flat = x.view(-1, x.shape[-1])
        x_fp8_flat = (x_flat / x_scale).to(torch.float8_e4m3fn)
        x_fp8 = x_fp8_flat.view(original_shape)
    else:
        # Quantize directly
        x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)

    return x_fp8, x_scale


def dequantize_fp8(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 tensor back to higher precision.

    Useful for debugging and verification.

    Args:
        x_fp8: FP8 tensor
        scale: Scale tensor used for quantization
        dtype: Target dtype (default: bfloat16)

    Returns:
        Dequantized tensor in target dtype
    """
    return x_fp8.to(dtype) * scale.to(dtype)


class FP8InputQuantizer:
    """Stateful FP8 input quantizer for repeated quantization operations.

    This class provides a convenient interface for quantizing inputs
    during inference, with configurable scale method and optional
    scale caching.
    """

    def __init__(
        self,
        scale_method: str = "per_token",
    ):
        """Initialize the quantizer.

        Args:
            scale_method: Scale computation method ("per_token" or "per_tensor")
        """
        self.scale_method = scale_method

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor.

        Args:
            x: Input tensor in BF16 or FP16

        Returns:
            Tuple of (FP8 tensor, scale tensor)
        """
        return quantize_input_to_fp8(x, scale_method=self.scale_method)


def prepare_fp8_gemm_input(
    x: torch.Tensor,
    scale_method: str = "per_token",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare input tensor for FP8 GEMM operation.

    This function:
    1. Reshapes input to 2D if needed
    2. Quantizes to FP8
    3. Ensures row-major layout (required by _scaled_mm)

    Args:
        x: Input tensor [batch, seq, hidden] or [M, K]
        scale_method: Scale computation method

    Returns:
        Tuple of:
            - x_fp8: FP8 tensor [M, K] in row-major layout
            - x_scale: Scale tensor [M, 1] or scalar
    """
    original_shape = x.shape

    # Flatten to 2D if needed
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    # Quantize
    x_fp8, x_scale = quantize_input_to_fp8(x, scale_method=scale_method)

    # Ensure contiguous (row-major)
    if not x_fp8.is_contiguous():
        x_fp8 = x_fp8.contiguous()

    return x_fp8, x_scale


__all__ = [
    "quantize_to_fp8",
    "compute_fp8_scale",
    "quantize_input_to_fp8",
    "dequantize_fp8",
    "FP8InputQuantizer",
    "prepare_fp8_gemm_input",
    "FP8_E4M3_MAX",
]