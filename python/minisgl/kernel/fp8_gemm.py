"""FP8 GEMM operations using Tensor Core.

This module provides FP8 matrix multiplication operations that leverage
hardware FP8 Tensor Core support (sm_89+).

Key requirements for torch._scaled_mm:
- mat1 (A): [M, K] row-major (stride = [K, 1])
- mat2 (B): [K, N] column-major (stride = [1, K])
- Computes: C = A @ B
- Scales are applied internally
"""

from __future__ import annotations

from typing import Tuple

import torch

from .input_quant import FP8_E4M3_MAX


def _ensure_column_major(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in column-major layout.

    For _scaled_mm, B needs to be column-major [K, N].
    This is achieved by transposing a row-major [N, K] tensor.

    Args:
        tensor: [K, N] tensor (will be viewed as column-major)

    Returns:
        Column-major view of the tensor
    """
    if tensor.stride()[0] == 1:
        # Already column-major
        return tensor
    # Transpose to get column-major view
    return tensor.T


def fp8_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """FP8 × FP8 GEMM using Tensor Core.

    Computes: output = (a * a_scale) @ (b * b_scale) + bias

    Args:
        a: Input activation [M, K] in FP8, row-major
        a_scale: Scale for a, either scalar or [M, 1]
        b: Weight matrix [K, N] in FP8, should be column-major
        b_scale: Scale for b, either scalar or block-wise
        bias: Optional bias tensor [N]
        out_dtype: Output data type (float16, bfloat16, float32)

    Returns:
        Output tensor [M, N] in out_dtype

    Note:
        For best performance, b should be pre-arranged in column-major format.
        This can be done by storing weights as [N, K] row-major, then using .T
        to get column-major [K, N] view.
    """
    # Ensure B is column-major
    if b.stride()[0] != 1:
        # B is row-major, transpose to column-major
        b = b.T

    # Run FP8 GEMM
    output = torch._scaled_mm(
        a,
        b,
        scale_a=a_scale,
        scale_b=b_scale,
        out_dtype=out_dtype,
    )

    # Add bias if present
    if bias is not None:
        output = output + bias

    return output


def fp8_gemm_with_block_scale(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: int = 128,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """FP8 GEMM with block-wise weight scale handling.

    The weight scale in minisgl is block-wise: [N//128, K//128].
    This function handles the scale broadcasting appropriately.

    Args:
        a: Input activation [M, K] in FP8, row-major
        a_scale: Per-token scale [M, 1] or scalar
        b: Weight matrix [K, N] in FP8
        b_scale: Block-wise scale [N//block_size, K//block_size]
        block_size: Block size for weight scale (default: 128)
        bias: Optional bias tensor [N]
        out_dtype: Output data type

    Returns:
        Output tensor [M, N] in out_dtype

    Note:
        Block-wise scale is currently handled by using a single max scale
        for the weight tensor. Full block-wise support would require
        custom CUDA kernels or tiling.
    """
    K, N = b.shape

    # For block-wise scale, we have two options:
    # 1. Use max scale (simpler but less precise)
    # 2. Tile the GEMM and apply different scales per block (more complex)

    # Option 1: Use max scale (current implementation)
    if b_scale.numel() > 1:
        # Block-wise scale - use max for simplicity
        # TODO: Implement tiled GEMM for full block-wise support
        b_scale_scalar = b_scale.amax().unsqueeze(0).float()  # Convert to float32 scalar
    else:
        b_scale_scalar = b_scale.float() if b_scale.dtype != torch.float32 else b_scale

    return fp8_gemm(a, a_scale, b, b_scale_scalar, bias, out_dtype)


def prepare_weight_for_fp8_gemm(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare FP8 weight for GEMM operation.

    This function:
    1. Ensures weight is in correct layout for _scaled_mm
    2. Returns weight and scale ready for GEMM

    Args:
        weight_fp8: Weight tensor [N, K] or [K, N] in FP8
        weight_scale: Scale tensor

    Returns:
        Tuple of (prepared weight, prepared scale)
    """
    # Weight should be [K, N] for column-major access
    # If weight is [N, K], transpose it
    if weight_fp8.shape[0] > weight_fp8.shape[1]:
        # Likely [N, K] where N > K
        weight_fp8 = weight_fp8.T

    # For _scaled_mm, we need column-major B
    # This means we store as [N, K] row-major, then view as [K, N] column-major
    # But if we already have [K, N], we just need to ensure column-major stride

    return weight_fp8, weight_scale


class FP8GEMM:
    """FP8 GEMM operator with cached weight layout."""

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.float16,
    ):
        """Initialize FP8 GEMM operator.

        Args:
            weight_fp8: Weight in FP8 format
            weight_scale: Weight scale
            bias: Optional bias
            out_dtype: Output data type
        """
        self.weight_fp8 = weight_fp8
        self.weight_scale = weight_scale
        self.bias = bias
        self.out_dtype = out_dtype

        # Pre-prepare weight for GEMM
        # Weight should be [K, N] column-major
        K, N = weight_fp8.shape
        if weight_fp8.stride()[0] != 1:
            # Need column-major - create by transposing [N, K]
            self.weight_col_major = weight_fp8.T
        else:
            self.weight_col_major = weight_fp8

    def __call__(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Run FP8 GEMM.

        Args:
            a: Input activation [M, K] in FP8
            a_scale: Input scale

        Returns:
            Output [M, N]
        """
        return fp8_gemm(
            a,
            a_scale,
            self.weight_col_major,
            self.weight_scale,
            self.bias,
            self.out_dtype,
        )


def convert_bf16_weight_to_fp8(
    weight: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert BF16 weight to FP8 format.

    Args:
        weight: BF16 weight tensor [N, K] or [K, N]
        block_size: Quantization block size

    Returns:
        Tuple of (FP8 weight, block-wise scale)
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got {weight.dim()}D")

    O, I = weight.shape

    # Pad to block size
    O_padded = ((O + block_size - 1) // block_size) * block_size
    I_padded = ((I + block_size - 1) // block_size) * block_size

    weight_padded = torch.zeros(O_padded, I_padded, dtype=weight.dtype, device=weight.device)
    weight_padded[:O, :I] = weight

    # Reshape for block-wise max
    weight_blocks = weight_padded.view(
        O_padded // block_size, block_size,
        I_padded // block_size, block_size
    )

    # Compute per-block max
    block_max = weight_blocks.abs().amax(dim=(1, 3))  # [O_blocks, I_blocks]
    block_scale = block_max / FP8_E4M3_MAX
    block_scale = block_scale.clamp(min=1e-12)

    # Quantize
    scale_expanded = block_scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    weight_fp8 = (weight_padded / scale_expanded).to(torch.float8_e4m3fn)

    # Return original shape
    weight_fp8 = weight_fp8[:O, :I]

    return weight_fp8, block_scale.to(torch.bfloat16)


__all__ = [
    "fp8_gemm",
    "fp8_gemm_with_block_scale",
    "FP8GEMM",
    "prepare_weight_for_fp8_gemm",
    "convert_bf16_weight_to_fp8",
]
