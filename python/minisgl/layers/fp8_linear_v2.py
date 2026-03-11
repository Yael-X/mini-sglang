"""FP8 Linear layers with Input Quantization.

This module provides FP8 Linear layers that use input quantization
instead of weight dequantization, enabling native FP8×FP8 Tensor Core GEMM.

Architecture:
- Weight: Stored as FP8 (same as existing FP8 layers)
- Input: Quantized to FP8 dynamically during forward
- GEMM: FP8×FP8 → FP16/FP32 using torch._scaled_mm

Layer Precision Strategy:
- QKV Projection: Keep BF16 (RoPE requires BF16)
- MLP (gate_up/down): Use FP8 input quantization (best fit for GEMM)
- Output Projection: Keep BF16 (precision sensitive)
"""

from __future__ import annotations

from typing import List

import torch
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.kernel import fp8_gemm, quantize_input_to_fp8
from minisgl.utils import div_even

from .base import BaseOP


class _Fp8InputQuantMixin:
    """Mixin class for FP8 Linear layers with input quantization.

    Unlike _Fp8LinearMixin which dequantizes weights to BF16,
    this mixin quantizes inputs to FP8 for native FP8 GEMM.
    """

    def _init_fp8_weights_v2(
        self, local_osize: int, local_isize: int, has_bias: bool, block_size: int = 128
    ) -> None:
        """Initialize FP8 weight storage (same as existing FP8 layers).

        Args:
            local_osize: Local output dimension size
            local_isize: Local input dimension size
            has_bias: Whether to allocate bias
            block_size: FP8 quantization block size (default: 128)
        """
        # Weight stored as FP8 [out_features, in_features]
        self.weight_fp8 = torch.empty(
            local_osize, local_isize, dtype=torch.float8_e4m3fn
        )
        # Block-wise scale [out_features//128, in_features//128]
        self.weight_scale = torch.empty(
            (local_osize + block_size - 1) // block_size,
            (local_isize + block_size - 1) // block_size,
            dtype=torch.bfloat16,
        )
        self.bias = torch.empty(local_osize) if has_bias else None
        # Clear parent's weight
        self.weight = None  # type: ignore

    def _forward_fp8_input_quant(
        self,
        x: torch.Tensor,
        scale_method: str = "per_tensor",
    ) -> torch.Tensor:
        """Forward pass with input quantization.

        Args:
            x: Input tensor [batch, seq, hidden] or [M, K]
            scale_method: "per_token" or "per_tensor"

        Returns:
            Output tensor
        """
        original_shape = x.shape

        # Flatten to 2D if needed
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        M, K = x.shape
        N = self.weight_fp8.shape[0]

        # 1. Quantize input to FP8
        x_fp8, x_scale = quantize_input_to_fp8(x, scale_method=scale_method)

        # 2. Prepare weight for GEMM
        # weight_fp8 is [N, K], need column-major [K, N] for _scaled_mm
        # Using .T gives column-major view
        weight_col = self.weight_fp8.T  # [K, N] column-major

        # 3. Get weight scale (convert to float32 for _scaled_mm, ensure on CUDA)
        # For block-wise scale, use max for now
        if self.weight_scale.numel() > 1:
            w_scale = self.weight_scale.amax().float().to(x.device)
        else:
            w_scale = self.weight_scale.float().to(x.device)

        # 4. FP8 GEMM
        output = fp8_gemm(
            x_fp8,
            x_scale,
            weight_col,
            w_scale,
            bias=None,  # Add bias separately
            out_dtype=torch.float16,
        )

        # 5. Add bias if present (in higher precision)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype).to(x.device)

        # 6. Reshape back to original shape if needed
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], N)

        return output

    def _forward_fp8_input_quant_row_parallel(
        self,
        x: torch.Tensor,
        scale_method: str = "per_tensor",
    ) -> torch.Tensor:
        """Forward pass for row-parallel with all_reduce."""
        output = self._forward_fp8_input_quant(x, scale_method)
        if self._tp_size > 1:
            output = self._comm.all_reduce(output)
        return output


class Fp8LinearColParallelMergedV2(_Fp8InputQuantMixin, BaseOP):
    """FP8 Linear layer with column-parallel merged weights and input quantization.

    Used for gate_up_proj in MLP layers.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        input_scale_method: str = "per_tensor",
    ):
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)

        self.full_input_size = input_size
        self.full_output_size = output_size
        self.local_input_size = input_size
        self.local_output_size = tp_output_size
        self.input_scale_method = input_scale_method

        self._init_fp8_weights_v2(tp_output_size, input_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fp8_input_quant(x, self.input_scale_method)


class Fp8LinearRowParallelV2(_Fp8InputQuantMixin, BaseOP):
    """FP8 Linear layer with row-parallel weights, input quantization, and all_reduce.

    Used for down_proj in MLP layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        input_scale_method: str = "per_tensor",
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size

        self.full_input_size = input_size
        self.full_output_size = output_size
        self.local_input_size = local_input_size
        self.local_output_size = local_output_size
        self.input_scale_method = input_scale_method

        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size

        self._init_fp8_weights_v2(local_output_size, local_input_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fp8_input_quant_row_parallel(x, self.input_scale_method)


# ======================= Mixed Precision MLP =======================


class Fp8GatedMLPWithInputQuant(BaseOP):
    """Gated MLP with FP8 input quantization for both projections.

    This is the main MLP implementation for FP8 inference with input quantization.
    It uses FP8 GEMM for both gate_up_proj and down_proj.

    Architecture:
        Input (BF16) → Quantize → FP8 GEMM (gate_up) → SiLU × Gate → Quantize → FP8 GEMM (down) → Output (FP16)

    Note:
        The intermediate activation between gate_up and down_proj is quantized
        again to FP8 before the second GEMM.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        has_bias: bool = False,
        input_scale_method: str = "per_tensor",
    ):
        from minisgl.layers import silu_and_mul, gelu_and_mul

        self.gate_up_proj = Fp8LinearColParallelMergedV2(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            has_bias=has_bias,
            input_scale_method=input_scale_method,
        )

        self.down_proj = Fp8LinearRowParallelV2(
            input_size=intermediate_size,
            output_size=hidden_size,
            has_bias=has_bias,
            input_scale_method=input_scale_method,
        )

        # Activation function
        act_fns = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        if hidden_act not in act_fns:
            raise ValueError(f"Unsupported activation: {hidden_act}")
        self.act_fn = act_fns[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, hidden] BF16

        # 1. gate_up projection (FP8 GEMM)
        gate_up = self.gate_up_proj.forward(x)  # [batch, seq, 2*intermediate] FP16
        del x

        # 2. Activation (SiLU/GELU) - operates in FP16
        y = self.act_fn(gate_up)  # [batch, seq, intermediate] FP16
        del gate_up

        # 3. down projection (FP8 GEMM)
        # Note: y is quantized inside down_proj.forward()
        output = self.down_proj.forward(y)  # [batch, seq, hidden] FP16

        return output


# ======================= Factory Functions =======================


def create_fp8_mlp_with_input_quant(
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str = "silu",
    has_bias: bool = False,
    input_scale_method: str = "per_tensor",
) -> Fp8GatedMLPWithInputQuant:
    """Create an FP8 MLP with input quantization.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: MLP intermediate dimension
        hidden_act: Activation function ("silu" or "gelu")
        has_bias: Whether to use bias
        input_scale_method: Scale method for input quantization
            - "per_tensor": Single scale for entire tensor (faster)
            - "per_token": Per-token scale (more precise)

    Returns:
        Fp8GatedMLPWithInputQuant instance
    """
    return Fp8GatedMLPWithInputQuant(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        has_bias=has_bias,
        input_scale_method=input_scale_method,
    )


__all__ = [
    "Fp8LinearColParallelMergedV2",
    "Fp8LinearRowParallelV2",
    "Fp8GatedMLPWithInputQuant",
    "create_fp8_mlp_with_input_quant",
]