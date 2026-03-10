from __future__ import annotations

import pytest
import torch

from minisgl.kernel.fp8 import Fp8DequantBuffer


@pytest.fixture(autouse=True)
def clear_fp8_dequant_buffer_instances():
    Fp8DequantBuffer.clear_all()
    yield
    Fp8DequantBuffer.clear_all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_dequant_buffer_keyed_by_device_and_stream():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    inst0 = Fp8DequantBuffer.get_instance(device0)
    inst1 = Fp8DequantBuffer.get_instance(device1)

    # Regression: different CUDA devices in the same process must not share one instance.
    assert inst0 is not inst1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_dequant_buffer_reuses_instance_on_same_device_stream():
    device0 = torch.device("cuda:0")

    a = Fp8DequantBuffer.get_instance(device0)
    b = Fp8DequantBuffer.get_instance(device0)

    assert a is b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_get_buffer_runtime_device_check():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    inst0 = Fp8DequantBuffer.get_instance(device0)
    _ = inst0.get_buffer((16, 16), device0)

    with pytest.raises(AssertionError, match="device mismatch"):
        inst0.get_buffer((8, 8), device1)
