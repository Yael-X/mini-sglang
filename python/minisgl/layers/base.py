from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]

_EXPERT_KEY_SUFFIXES = ("", ".weight", ".bias")


def _collect_expert_keys(
    state_dict: _STATE_DICT, prefix: str, param_name: str
) -> List[str]:
    """Collect expert weight keys in O(num_experts) via direct dict lookup."""
    keys: List[str] = []
    idx = 0
    while True:
        found = False
        for suffix in _EXPERT_KEY_SUFFIXES:
            candidate = f"{prefix}.{idx}.{param_name}{suffix}"
            if candidate in state_dict:
                keys.append(candidate)
                found = True
                break
        if not found:
            break
        idx += 1

    if keys:
        return keys

    # Fallback: linear scan for non-standard key naming conventions
    for key in list(state_dict.keys()):
        if prefix in key and param_name in key:
            keys.append(key)

    def _expert_index(k: str) -> int:
        match = re.search(r"experts\.(\d+)\.", k)
        return int(match.group(1)) if match else 0

    keys.sort(key=_expert_index)
    return keys


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    float8_names = ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    return any(hasattr(torch, name) and dtype == getattr(torch, name) for name in float8_names)


class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue

            # FP8 weight loading: handle weight_fp8 and weight_scale
            if name == "weight_fp8" and isinstance(param, torch.Tensor):
                weight_key = _concat_prefix(prefix, "weight")
                scale_key = f"{weight_key}_scale_inv"

                has_weight = weight_key in state_dict
                has_scale = scale_key in state_dict

                if has_weight and not has_scale:
                    raise ValueError(
                        f"Missing FP8 scale tensor for '{prefix}': expected key '{scale_key}' "
                        f"for weight key '{weight_key}'."
                    )

                if has_weight and has_scale:
                    loaded_weight_fp8 = state_dict.pop(weight_key)
                    loaded_weight_scale = state_dict.pop(scale_key)

                    expected_weight_shape = tuple(self.weight_fp8.shape)
                    expected_scale_shape = tuple(self.weight_scale.shape)
                    actual_weight_shape = tuple(loaded_weight_fp8.shape)
                    actual_scale_shape = tuple(loaded_weight_scale.shape)

                    if not _is_float8_dtype(loaded_weight_fp8.dtype):
                        raise ValueError(
                            f"Invalid FP8 weight dtype at '{prefix}': expected float8 dtype, "
                            f"got {loaded_weight_fp8.dtype}."
                        )
                    if loaded_weight_scale.dtype != torch.bfloat16:
                        raise ValueError(
                            f"Invalid FP8 scale dtype at '{prefix}': expected torch.bfloat16, "
                            f"got {loaded_weight_scale.dtype}."
                        )

                    if actual_weight_shape != expected_weight_shape:
                        raise ValueError(
                            f"Invalid FP8 weight shape at '{prefix}': expected {expected_weight_shape}, "
                            f"got {actual_weight_shape}."
                        )

                    expected_scale_from_block = (
                        (actual_weight_shape[0] + 127) // 128,
                        (actual_weight_shape[1] + 127) // 128,
                    )

                    if (
                        actual_scale_shape != expected_scale_shape
                        or actual_scale_shape != expected_scale_from_block
                    ):
                        raise ValueError(
                            f"Invalid FP8 scale shape at '{prefix}': expected {expected_scale_shape} "
                            f"(block-128 from weight -> {expected_scale_from_block}), "
                            f"got {actual_scale_shape}."
                        )

                    self.weight_fp8 = loaded_weight_fp8
                    self.weight_scale = loaded_weight_scale
                    continue

            # Skip weight_scale when it's part of FP8 (already loaded with weight_fp8)
            if name == "weight_scale" and "weight_fp8" in self.__dict__:
                continue

            if isinstance(param, torch.Tensor):
                if "experts" in prefix:
                    mapped_name = name
                    matched_keys = _collect_expert_keys(state_dict, prefix, mapped_name)

                    items = [state_dict.pop(k) for k in matched_keys]
                    if not items:
                        raise ValueError(
                            f"No weights found in state_dict for {prefix} and {mapped_name}"
                        )

                    item = torch.stack(items, dim=0)
                else:
                    item = state_dict.pop(_concat_prefix(prefix, name))

                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype

                setattr(self, name, item)

            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            _ = prefix
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        _ = prefix
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
