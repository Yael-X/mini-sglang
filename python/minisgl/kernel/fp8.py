"""FP8 dequantization kernel and buffer management."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def dequantize_fp8_block(
    weight: torch.Tensor,  # [O, I] float8_e4m3fn
    scale: torch.Tensor,  # [O//128, I//128] bfloat16
    output: torch.Tensor,  # [O, I] bfloat16 (pre-allocated)
    block_size: tuple[int, int] = (128, 128),
) -> None:
    """
    In-place FP8 -> BF16 dequantization to pre-allocated buffer.

    Uses view + broadcast mechanism to avoid extra memory allocation from repeat_interleave.

    Args:
        weight: FP8 weight tensor [out_features, in_features]
        scale: Scale tensor [out_features//block_size[0], in_features//block_size[1]]
        output: Pre-allocated output buffer [out_features, in_features]
        block_size: Size of each quantization block (default: 128x128)

    Performance comparison:
        - repeat_interleave: Creates ~250MB extra allocation for 4096x4096 weight
        - view + broadcast: 0 extra memory allocation, computation fused in broadcast
    """
    O, I = weight.shape
    bR, bC = block_size

    main_rows = (O // bR) * bR
    main_cols = (I // bC) * bC

    # Fast path: complete block region using view+broadcast (no extra allocation)
    if main_rows > 0 and main_cols > 0:
        row_blocks = main_rows // bR
        col_blocks = main_cols // bC
        w_view = weight[:main_rows, :main_cols].view(row_blocks, bR, col_blocks, bC)
        s_view = scale[:row_blocks, :col_blocks].view(row_blocks, 1, col_blocks, 1)
        output[:main_rows, :main_cols].view(row_blocks, bR, col_blocks, bC).copy_(
            w_view.to(torch.bfloat16) * s_view.to(torch.bfloat16)
        )

    # Tail rows: [main_rows:O, :main_cols]
    if main_rows < O and main_cols > 0:
        row_scale_idx = main_rows // bR
        col_blocks = main_cols // bC
        tail_scale = scale[row_scale_idx : row_scale_idx + 1, :col_blocks].to(torch.bfloat16)
        tail_scale = tail_scale.repeat_interleave(bC, dim=1)[:, :main_cols]
        output[main_rows:O, :main_cols].copy_(
            weight[main_rows:O, :main_cols].to(torch.bfloat16) * tail_scale
        )

    # Tail cols: [:main_rows, main_cols:I]
    if main_cols < I and main_rows > 0:
        col_scale_idx = main_cols // bC
        row_blocks = main_rows // bR
        tail_scale = scale[:row_blocks, col_scale_idx : col_scale_idx + 1].to(torch.bfloat16)
        tail_scale = tail_scale.repeat_interleave(bR, dim=0)[:main_rows]
        output[:main_rows, main_cols:I].copy_(
            weight[:main_rows, main_cols:I].to(torch.bfloat16) * tail_scale
        )

    # Bottom-right tail corner: [main_rows:O, main_cols:I]
    if main_rows < O and main_cols < I:
        row_scale_idx = main_rows // bR
        col_scale_idx = main_cols // bC
        corner_scale = scale[row_scale_idx : row_scale_idx + 1, col_scale_idx : col_scale_idx + 1]
        output[main_rows:O, main_cols:I].copy_(
            weight[main_rows:O, main_cols:I].to(torch.bfloat16)
            * corner_scale.to(torch.bfloat16)
        )


class Fp8DequantBuffer:
    """
    FP8 dequantization buffer manager.

    Design decisions:
    - Allocate independent buffer per (CUDA device, CUDA Stream) to avoid cross-device
      and cross-stream data races.
    - Lazy allocation on first use
    - Auto-expansion as needed

    Lifecycle / cleanup:
    - Instances live in a process-global cache (`_instances`) for reuse across forwards.
    - Memory is released by `clear_all()`, which is currently used by tests and can also be
      called by upper-layer teardown hooks when unloading a model.
    - If `clear_all()` is not called explicitly, cleanup still happens at process exit.
    """

    _instances: Dict[Tuple[int, int], "Fp8DequantBuffer"] = {}  # (device_index, stream_id)

    def __init__(self, device: torch.device):
        self.device = device
        self._buffer: torch.Tensor | None = None
        self._max_shape: tuple[int, int] = (0, 0)

    @classmethod
    def get_instance(cls, device: torch.device) -> "Fp8DequantBuffer":
        """Get the buffer instance for the specified CUDA device and current stream."""
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"Fp8DequantBuffer only supports CUDA device, got {device}")

        # Explicitly query stream on the requested device to avoid accidental cross-device reuse.
        stream = torch.cuda.current_stream(device=device)
        stream_id = int(stream.cuda_stream)
        device_index = torch.cuda._utils._get_device_index(device, optional=False)
        key = (device_index, stream_id)
        if key not in cls._instances:
            cls._instances[key] = cls(device)
        return cls._instances[key]

    def get_buffer(self, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        """
        Get a buffer of at least the specified size, expanding if necessary.

        Args:
            shape: (output_dim, input_dim)
            device: Expected CUDA device for this request

        Returns:
            Pre-allocated BF16 buffer, size >= shape
        """
        if (
            self._buffer is None
            or self._max_shape[0] < shape[0]
            or self._max_shape[1] < shape[1]
        ):
            self._max_shape = (
                max(self._max_shape[0], shape[0]),
                max(self._max_shape[1], shape[1]),
            )
            self._buffer = torch.empty(
                self._max_shape, dtype=torch.bfloat16, device=self.device
            )
        device = torch.device(device)
        assert self._buffer is not None
        assert self._buffer.device == device, (
            f"Fp8DequantBuffer device mismatch: buffer={self._buffer.device}, request={device}"
        )
        return self._buffer[: shape[0], : shape[1]]

    @classmethod
    def clear_all(cls) -> None:
        """Clear all buffers (for memory cleanup)."""
        cls._instances.clear()

    @classmethod
    def get_max_shape(cls) -> tuple[int, int]:
        """Get the maximum shape across all instances."""
        max_shape = (0, 0)
        for instance in cls._instances.values():
            if instance._max_shape[0] > max_shape[0] or instance._max_shape[1] > max_shape[1]:
                max_shape = instance._max_shape
        return max_shape
