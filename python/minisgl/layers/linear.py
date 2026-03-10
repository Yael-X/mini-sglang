from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class _Fp8LinearMixin:
    """Mixin class for FP8 Linear layers providing FP8 weight management."""

    def _init_fp8_weights(
        self, local_osize: int, local_isize: int, has_bias: bool, block_size: int = 128
    ) -> None:
        """Initialize FP8 weight storage.

        Args:
            local_osize: Local output dimension size
            local_isize: Local input dimension size
            has_bias: Whether to allocate bias
            block_size: FP8 quantization block size (default: 128)
        """
        self.weight_fp8 = torch.empty(
            local_osize, local_isize, dtype=torch.float8_e4m3fn
        )
        self.weight_scale = torch.empty(
            (local_osize + block_size - 1) // block_size,
            (local_isize + block_size - 1) // block_size,
            dtype=torch.bfloat16,
        )
        self.bias = torch.empty(local_osize) if has_bias else None
        # Clear parent's weight to avoid duplicate storage
        self.weight = None  # type: ignore

    def _forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly FP8 dequantization.

        Args:
            x: Input tensor

        Returns:
            Output tensor after FP8 linear operation
        """
        from minisgl.kernel.fp8 import Fp8DequantBuffer, dequantize_fp8_block

        buffer = Fp8DequantBuffer.get_instance(x.device)
        dequant_weight = buffer.get_buffer(
            (self.weight_fp8.shape[0], self.weight_fp8.shape[1]), x.device
        )
        dequantize_fp8_block(self.weight_fp8, self.weight_scale, dequant_weight)
        return F.linear(x, dequant_weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = div_even(num_qo_heads, num_kv_heads)
        local_num_kv = div_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


# ======================= FP8 Linear Layers =======================


class Fp8LinearReplicated(_Fp8LinearMixin, _LinearTPImpl):
    """FP8 Linear layer with replicated (non-sharded) weights."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        # Initialize parent with full sizes (no sharding)
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )
        # Override with FP8 weights
        self._init_fp8_weights(output_size, input_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fp8(x)


class Fp8LinearColParallelMerged(_Fp8LinearMixin, _LinearTPImpl):
    """FP8 Linear layer with column-parallel merged weights (e.g., gate_up_proj)."""

    def __init__(self, input_size: int, output_sizes: List[int], has_bias: bool):
        tp_info = get_tp_info()
        # Check that all output sizes are divisible by tp_size
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)

        # Initialize parent
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)
        # Override with FP8 weights
        self._init_fp8_weights(tp_output_size, input_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fp8(x)


class Fp8LinearQKVMerged(_Fp8LinearMixin, _LinearTPImpl):
    """FP8 Linear layer for merged QKV projection."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = div_even(num_qo_heads, num_kv_heads)
        local_num_kv = div_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim

        # Initialize parent
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)
        # Override with FP8 weights
        self._init_fp8_weights(local_osize, local_isize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fp8(x)


class Fp8LinearOProj(_Fp8LinearMixin, _LinearTPImpl):
    """FP8 Linear layer for output projection with all_reduce."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size

        # Initialize parent
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)
        # Override with FP8 weights
        self._init_fp8_weights(local_osize, local_isize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._forward_fp8(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class Fp8LinearRowParallel(_Fp8LinearMixin, _LinearTPImpl):
    """FP8 Linear layer with row-parallel weights and all_reduce."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size

        # Initialize parent
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)
        # Override with FP8 weights
        self._init_fp8_weights(local_output_size, local_input_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._forward_fp8(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
