from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import detect_quant_method, load_weight


def create_model(
    model_config: ModelConfig,
    use_fp8: bool = False,
    use_fp8_input_quant: bool = False,
    fp8_input_scale_method: str = "per_tensor",
) -> BaseLLMModel:
    return get_model_class(
        model_config.architectures[0],
        model_config,
        use_fp8=use_fp8,
        use_fp8_input_quant=use_fp8_input_quant,
        fp8_input_scale_method=fp8_input_scale_method,
    )


__all__ = ["create_model", "detect_quant_method", "load_weight", "RotaryConfig"]
