"""Tests for FP8 Linear layers with input quantization."""

from __future__ import annotations

import pytest
import torch

from minisgl.layers.fp8_linear_v2 import (
    Fp8GatedMLPWithInputQuant,
    Fp8LinearColParallelMergedV2,
    Fp8LinearRowParallelV2,
    create_fp8_mlp_with_input_quant,
)


# Session-scoped fixture to set TP info once
@pytest.fixture(scope="session", autouse=True)
def setup_tp_info_session():
    """Set up TP info once for the test session."""
    from minisgl.distributed import set_tp_info
    from minisgl.distributed.info import _TP_INFO

    if _TP_INFO is None:
        set_tp_info(0, 1)  # Single GPU


def random_fp8_tensor(shape: tuple, device: str = "cuda") -> torch.Tensor:
    """Create a random FP8 tensor by converting from BF16."""
    return torch.randn(shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFp8LinearColParallelMergedV2:
    """Tests for Fp8LinearColParallelMergedV2."""

    def test_init(self) -> None:
        """Test layer initialization."""
        layer = Fp8LinearColParallelMergedV2(
            input_size=256,
            output_sizes=[512, 512],
            has_bias=False,
        )

        assert layer.weight_fp8.shape == (1024, 256)  # Merged output
        assert layer.weight_fp8.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape[0] == (1024 + 127) // 128
        assert layer.weight_scale.shape[1] == (256 + 127) // 128

    def test_forward_shape(self) -> None:
        """Test forward output shape."""
        layer = Fp8LinearColParallelMergedV2(
            input_size=128,
            output_sizes=[64, 64],
            has_bias=False,
        )

        # Initialize with random weights (via BF16 conversion)
        layer.weight_fp8 = random_fp8_tensor(layer.weight_fp8.shape)
        layer.weight_scale = torch.ones_like(layer.weight_scale)

        x = torch.randn(2, 16, 128, dtype=torch.bfloat16, device="cuda")
        output = layer.forward(x)

        assert output.shape == (2, 16, 128)  # Combined output: 64 + 64 = 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFp8LinearRowParallelV2:
    """Tests for Fp8LinearRowParallelV2."""

    def test_init(self) -> None:
        """Test layer initialization."""
        layer = Fp8LinearRowParallelV2(
            input_size=256,
            output_size=128,
            has_bias=False,
        )

        assert layer.weight_fp8.dtype == torch.float8_e4m3fn
        assert layer.bias is None

    def test_forward_shape(self) -> None:
        """Test forward output shape."""
        layer = Fp8LinearRowParallelV2(
            input_size=128,
            output_size=64,
            has_bias=False,
        )

        # Initialize with random weights (via BF16 conversion)
        layer.weight_fp8 = random_fp8_tensor(layer.weight_fp8.shape)
        layer.weight_scale = torch.ones_like(layer.weight_scale)

        x = torch.randn(2, 16, 128, dtype=torch.bfloat16, device="cuda")
        output = layer.forward(x)

        assert output.shape == (2, 16, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFp8GatedMLPWithInputQuant:
    """Tests for Fp8GatedMLPWithInputQuant."""

    def test_init(self) -> None:
        """Test MLP initialization."""
        mlp = Fp8GatedMLPWithInputQuant(
            hidden_size=256,
            intermediate_size=512,
            hidden_act="silu",
            has_bias=False,
        )

        assert mlp.gate_up_proj is not None
        assert mlp.down_proj is not None

    def test_forward_shape(self) -> None:
        """Test forward output shape."""
        mlp = Fp8GatedMLPWithInputQuant(
            hidden_size=128,
            intermediate_size=256,
            hidden_act="silu",
            has_bias=False,
        )

        # Initialize weights
        mlp.gate_up_proj.weight_fp8 = random_fp8_tensor(mlp.gate_up_proj.weight_fp8.shape)
        mlp.gate_up_proj.weight_scale = torch.ones_like(mlp.gate_up_proj.weight_scale)
        mlp.down_proj.weight_fp8 = random_fp8_tensor(mlp.down_proj.weight_fp8.shape)
        mlp.down_proj.weight_scale = torch.ones_like(mlp.down_proj.weight_scale)

        x = torch.randn(2, 16, 128, dtype=torch.bfloat16, device="cuda")
        output = mlp.forward(x)

        assert output.shape == (2, 16, 128)  # Same as hidden_size

    def test_forward_with_different_hidden_acts(self) -> None:
        """Test forward with different activation functions."""
        for act in ["silu", "gelu"]:
            mlp = Fp8GatedMLPWithInputQuant(
                hidden_size=64,
                intermediate_size=128,
                hidden_act=act,
                has_bias=False,
            )

            # Initialize weights
            mlp.gate_up_proj.weight_fp8 = random_fp8_tensor(mlp.gate_up_proj.weight_fp8.shape)
            mlp.gate_up_proj.weight_scale = torch.ones_like(mlp.gate_up_proj.weight_scale)
            mlp.down_proj.weight_fp8 = random_fp8_tensor(mlp.down_proj.weight_fp8.shape)
            mlp.down_proj.weight_scale = torch.ones_like(mlp.down_proj.weight_scale)

            x = torch.randn(1, 8, 64, dtype=torch.bfloat16, device="cuda")
            output = mlp.forward(x)

            assert output.shape == (1, 8, 64), f"Failed for activation {act}"

    def test_forward_with_bias(self) -> None:
        """Test forward with bias."""
        mlp = Fp8GatedMLPWithInputQuant(
            hidden_size=64,
            intermediate_size=128,
            hidden_act="silu",
            has_bias=True,
        )

        # Initialize weights and bias
        mlp.gate_up_proj.weight_fp8 = random_fp8_tensor(mlp.gate_up_proj.weight_fp8.shape)
        mlp.gate_up_proj.weight_scale = torch.ones_like(mlp.gate_up_proj.weight_scale)
        mlp.gate_up_proj.bias = torch.zeros(mlp.gate_up_proj.local_output_size)

        mlp.down_proj.weight_fp8 = random_fp8_tensor(mlp.down_proj.weight_fp8.shape)
        mlp.down_proj.weight_scale = torch.ones_like(mlp.down_proj.weight_scale)
        mlp.down_proj.bias = torch.zeros(mlp.down_proj.local_output_size)

        x = torch.randn(1, 8, 64, dtype=torch.bfloat16, device="cuda")
        output = mlp.forward(x)

        assert output.shape == (1, 8, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestCreateFp8MlpWithInputQuant:
    """Tests for factory function."""

    def test_factory(self) -> None:
        """Test factory function."""
        mlp = create_fp8_mlp_with_input_quant(
            hidden_size=256,
            intermediate_size=512,
            hidden_act="silu",
            has_bias=False,
            input_scale_method="per_tensor",
        )

        assert isinstance(mlp, Fp8GatedMLPWithInputQuant)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestFp8LinearWeightLoading:
    """Tests for weight loading compatibility."""

    def test_weight_shape_compatibility(self) -> None:
        """Test that weight shapes are compatible with existing FP8 loading."""
        # Existing FP8 layer weights are stored as:
        # - weight_fp8: [out_features, in_features] float8_e4m3fn
        # - weight_scale: [out_features//128, in_features//128] bfloat16

        layer = Fp8LinearColParallelMergedV2(
            input_size=4096,
            output_sizes=[11008, 11008],  # Llama-like
            has_bias=False,
        )

        # Check weight shapes
        assert layer.weight_fp8.shape[0] == 22016  # 11008 * 2
        assert layer.weight_fp8.shape[1] == 4096

        # Check scale shapes (block_size = 128)
        assert layer.weight_scale.shape[0] == (22016 + 127) // 128
        assert layer.weight_scale.shape[1] == (4096 + 127) // 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])