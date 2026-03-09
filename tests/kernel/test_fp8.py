from __future__ import annotations

import torch

from minisgl.kernel.fp8 import dequantize_fp8_block
from minisgl.models.weight import _get_fp8_shard_bounds, _shard_fp8_scale


BLOCK = 128


def _reference_dequant(
    weight: torch.Tensor, scale: torch.Tensor, block_size: tuple[int, int] = (BLOCK, BLOCK)
) -> torch.Tensor:
    row_block, col_block = block_size
    expanded = scale.repeat_interleave(row_block, dim=0).repeat_interleave(col_block, dim=1)
    return weight.to(torch.bfloat16) * expanded[: weight.shape[0], : weight.shape[1]].to(torch.bfloat16)


def test_dequantize_fp8_block_supports_non_divisible_shapes() -> None:
    out_features = 257
    in_features = 385
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    scale = torch.rand((out_features + BLOCK - 1) // BLOCK, (in_features + BLOCK - 1) // BLOCK, dtype=torch.float32)
    output = torch.empty_like(weight, dtype=torch.bfloat16)

    dequantize_fp8_block(weight, scale, output, (BLOCK, BLOCK))
    expected = _reference_dequant(weight, scale)

    assert torch.equal(output, expected)


def test_fp8_weight_scale_shard_alignment_hidden_intermediate_vocab() -> None:
    tp_size = 3
    # hidden_size, intermediate_size, vocab_size intentionally not divisible by 128 * tp
    shape_cases = [
        ((3073, 769), 0),    # hidden-like projection split on output dim
        ((769, 5505), 1),    # intermediate-like projection split on input dim
        ((151937, 769), 0),  # vocab-like split on vocab/output dim
    ]

    for weight_shape, split_dim in shape_cases:
        scale_shape = (
            (weight_shape[0] + BLOCK - 1) // BLOCK,
            (weight_shape[1] + BLOCK - 1) // BLOCK,
        )
        scale = torch.arange(scale_shape[0] * scale_shape[1], dtype=torch.float32).view(scale_shape)

        for rank in range(tp_size):
            row_start, row_end, col_start, col_end = _get_fp8_shard_bounds(
                weight_shape, split_dim, rank, tp_size
            )
            sharded_scale = _shard_fp8_scale(
                scale,
                row_start,
                row_end,
                col_start,
                col_end,
                BLOCK,
            )

            expected = scale[
                row_start // BLOCK : (row_end + BLOCK - 1) // BLOCK,
                col_start // BLOCK : (col_end + BLOCK - 1) // BLOCK,
            ]
            assert torch.equal(sharded_scale, expected)
