from __future__ import annotations

import pytest
import torch

from minisgl.layers.base import BaseOP


class _Fp8Leaf(BaseOP):
    def __init__(self):
        self.weight_fp8 = torch.empty(128, 128, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(1, 1, dtype=torch.bfloat16)

    def forward(self, *args, **kwargs):
        return None


class _Root(BaseOP):
    def __init__(self):
        self.layer = _Fp8Leaf()

    def forward(self, *args, **kwargs):
        return None


def test_fp8_load_state_dict_raises_on_invalid_scale_shape_with_layer_prefix():
    root = _Root()
    state_dict = {
        "layer.weight": torch.empty(128, 128, dtype=torch.float8_e4m3fn),
        "layer.weight_scale_inv": torch.empty(2, 1, dtype=torch.bfloat16),
    }

    with pytest.raises(ValueError, match=r"layer") as exc_info:
        root.load_state_dict(state_dict)

    msg = str(exc_info.value)
    assert "expected (1, 1)" in msg
    assert "got (2, 1)" in msg


def test_fp8_load_state_dict_raises_on_invalid_scale_dtype_with_layer_prefix():
    root = _Root()
    state_dict = {
        "layer.weight": torch.empty(128, 128, dtype=torch.float8_e4m3fn),
        "layer.weight_scale_inv": torch.empty(1, 1, dtype=torch.float16),
    }

    with pytest.raises(ValueError, match=r"layer") as exc_info:
        root.load_state_dict(state_dict)

    msg = str(exc_info.value)
    assert "expected torch.bfloat16" in msg
    assert "got torch.float16" in msg


def test_fp8_load_state_dict_raises_when_scale_missing():
    root = _Root()
    state_dict = {
        "layer.weight": torch.empty(128, 128, dtype=torch.float8_e4m3fn),
    }

    with pytest.raises(ValueError, match=r"layer") as exc_info:
        root.load_state_dict(state_dict)

    msg = str(exc_info.value)
    assert "Missing FP8 scale tensor" in msg
    assert "layer.weight_scale_inv" in msg
