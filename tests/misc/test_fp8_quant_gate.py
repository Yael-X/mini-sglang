from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from minisgl.engine.engine import Engine, _resolve_fp8_keep_quantized
from minisgl.models.weight import detect_quant_method


def test_detect_quant_method_reads_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}})
    )
    assert detect_quant_method(str(tmp_path)) == "fp8"


def test_bf16_model_with_keep_quantized_flag_is_downgraded(monkeypatch):
    monkeypatch.setattr("minisgl.engine.engine.detect_quant_method", lambda _: None)

    use_fp8, warning = _resolve_fp8_keep_quantized("dummy-model", requested_fp8_keep_quantized=True)

    assert use_fp8 is False
    assert warning is not None
    assert "only valid for FP8 models" in warning


def test_fp8_model_without_flag_uses_dequantized_loading(monkeypatch):
    monkeypatch.setattr("minisgl.engine.engine.detect_quant_method", lambda _: "fp8")
    use_fp8, warning = _resolve_fp8_keep_quantized("dummy-model", requested_fp8_keep_quantized=False)
    assert use_fp8 is False
    assert warning is None

    calls = []

    def fake_load_weight(model_path, device, fp8_keep_quantized):
        calls.append((model_path, device, fp8_keep_quantized))
        return {"w": torch.ones(1, dtype=torch.float16)}

    monkeypatch.setattr("minisgl.engine.engine.load_weight", fake_load_weight)

    engine = object.__new__(Engine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16

    config = SimpleNamespace(use_dummy_weight=False, model_path="dummy-model")
    state_dict = Engine._load_weight_state_dict(engine, config, use_fp8=use_fp8)

    assert calls == [("dummy-model", torch.device("cpu"), False)]
    assert state_dict["w"].dtype == torch.bfloat16
