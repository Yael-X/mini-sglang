from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig, use_fp8: bool = False) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config, use_fp8)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
