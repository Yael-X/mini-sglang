from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import safetensors
import torch
from tqdm import tqdm
from minisgl.distributed import get_tp_info, try_get_tp_info
from minisgl.utils import div_ceil, download_hf_weight


def detect_quant_method(model_path: str) -> Optional[str]:
    """Detect quantization method from ``config.json``.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        The value of ``quantization_config.quant_method`` if available,
        otherwise ``None``.
    """
    model_folder = download_hf_weight(model_path)
    config_path = os.path.join(model_folder, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        config = json.load(f)
    quant_config = config.get("quantization_config", {})
    return quant_config.get("quant_method")


def _is_fp8_model(model_folder: str) -> bool:
    """Check if the model is FP8 quantized by examining config.json."""
    return detect_quant_method(model_folder) == "fp8"


def _is_awq_model(model_folder: str) -> bool:
    """Check if the model is AWQ quantized by examining config.json."""
    return detect_quant_method(model_folder) == "awq"


def _dequantize_fp8_block(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: Tuple[int, int] = (128, 128),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize FP8 weights to BF16 using block-wise expansion.

    Args:
        weight: FP8 weight tensor [out_features, in_features]
        scale: Scale tensor [out_features//block_size[0], in_features//block_size[1]]
        block_size: Size of each quantization block
        dtype: Target dtype

    Returns:
        Dequantized weight tensor in target dtype
    """
    row_block, col_block = block_size
    out_features, in_features = weight.shape

    # Use repeat_interleave to expand scale to weight dimensions
    # This is more memory efficient than creating intermediate tensors
    scale_expanded = scale.repeat_interleave(row_block, dim=0).repeat_interleave(col_block, dim=1)

    # Trim to exact size
    scale_expanded = scale_expanded[:out_features, :in_features]

    # Dequantize: weight_fp8 * scale = weight_bf16
    return weight.to(dtype) * scale_expanded.to(dtype)


def _dequantize_fp8_linear(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: Tuple[int, int],
    split_dim: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize FP8 weights with tensor parallelism awareness.

    For TP, we need to dequantize first, then shard the dequantized weights.
    """
    # Dequantize the full weight
    weight_deq = _dequantize_fp8_block(weight, scale, block_size, dtype)

    # Apply tensor parallelism sharding
    if tp_size > 1:
        if split_dim == 0:
            # Split output dimension (e.g., q_proj, k_proj, v_proj, gate_proj, up_proj)
            out_features = weight_deq.shape[0]
            chunk_size = (out_features + tp_size - 1) // tp_size
            start_idx = tp_rank * chunk_size
            end_idx = min(start_idx + chunk_size, out_features)
            weight_deq = weight_deq[start_idx:end_idx]
        elif split_dim == 1:
            # Split input dimension (e.g., o_proj, down_proj)
            in_features = weight_deq.shape[1]
            chunk_size = (in_features + tp_size - 1) // tp_size
            start_idx = tp_rank * chunk_size
            end_idx = min(start_idx + chunk_size, in_features)
            weight_deq = weight_deq[:, start_idx:end_idx]

    return weight_deq


def _shard_fp8_weight(
    weight: torch.Tensor, split_dim: int, rank: int, size: int
) -> torch.Tensor:
    """Shard FP8 weight for tensor parallelism.

    Args:
        weight: FP8 weight tensor [out_features, in_features]
        split_dim: Dimension to split (0 for output, 1 for input)
        rank: TP rank
        size: TP world size

    Returns:
        Sharded FP8 weight tensor
    """
    if size == 1:
        return weight
    if split_dim == 0:
        out_features = weight.shape[0]
        chunk_size = (out_features + size - 1) // size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, out_features)
        return weight[start_idx:end_idx]
    else:
        in_features = weight.shape[1]
        chunk_size = (in_features + size - 1) // size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, in_features)
        return weight[:, start_idx:end_idx]


def _get_fp8_shard_bounds(
    weight_shape: Tuple[int, int], split_dim: int, rank: int, size: int
) -> Tuple[int, int, int, int]:
    """Get [row_start,row_end) and [col_start,col_end) shard bounds for one TP rank."""
    out_features, in_features = weight_shape
    if split_dim == 0:
        chunk_size = (out_features + size - 1) // size
        row_start = rank * chunk_size
        row_end = min(row_start + chunk_size, out_features)
        return row_start, row_end, 0, in_features

    chunk_size = (in_features + size - 1) // size
    col_start = rank * chunk_size
    col_end = min(col_start + chunk_size, in_features)
    return 0, out_features, col_start, col_end


def _shard_fp8_scale(
    scale: torch.Tensor,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Shard scale tensor for tensor parallelism.

    Note: scale dimensions are weight dimensions / block_size (block-wise quantization).
    - weight: [O, I] -> scale: [O//128, I//128]

    Args:
        scale: Scale tensor [out_features//block_size, in_features//block_size]
        row_start: Weight shard row start index
        row_end: Weight shard row end index
        col_start: Weight shard col start index
        col_end: Weight shard col end index
        block_size: FP8 quantization block size (default: 128)

    Returns:
        Sharded scale tensor
    """
    row_block_start = row_start // block_size
    row_block_end = div_ceil(row_end, block_size)
    col_block_start = col_start // block_size
    col_block_end = div_ceil(col_end, block_size)
    return scale[row_block_start:row_block_end, col_block_start:col_block_end]


def _load_fp8_weights(
    files: List[str],
    device: torch.device,
    tp_rank: int,
    tp_size: int,
    block_size: Tuple[int, int] = (128, 128),
) -> Dict[str, torch.Tensor]:
    """
    Load and dequantize FP8 weights.

    FP8 weights in Qwen3-8B-FP8 are stored as:
    - <layer>.<weight_name>: FP8 e4m3fn tensor
    - <layer>.<weight_name>_scale_inv: per-block scale tensor

    Note: We need to merge q/k/v and gate/up weights AND their scales first,
    then dequantize the merged weights.
    """
    state_dict: Dict[str, torch.Tensor] = {}
    scale_dict: Dict[str, torch.Tensor] = {}

    device_str = str(device)

    # First pass: load all weights and scales
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device=device_str) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                # Handle scale tensors (naming: xxx.weight_scale_inv)
                if name.endswith("_scale_inv"):
                    scale_dict[name] = tensor
                else:
                    state_dict[name] = tensor

    # Second pass: merge qkv and gate_up weights AND their scales
    state_dict, scale_dict = _merge_fp8_state_dict(state_dict, scale_dict)

    # Third pass: dequantize
    dequantized: Dict[str, torch.Tensor] = {}

    for name, weight in state_dict.items():
        # Check if this weight has a corresponding scale
        scale_name = f"{name}_scale_inv"
        if scale_name in scale_dict:
            scale = scale_dict[scale_name]

            # Determine split dimension for TP (already merged, so qkv_proj splits on dim 0)
            split_dim = 0
            if any(name.endswith(p) for p in [".o_proj", ".down_proj"]):
                split_dim = 1
            elif name.endswith("lm_head") or name.endswith("embed_tokens"):
                split_dim = 0  # Split by vocab dimension

            # Dequantize with TP awareness
            dequantized[name] = _dequantize_fp8_linear(
                weight, scale, block_size, split_dim, tp_rank, tp_size
            )
        else:
            # Non-quantized weights (e.g., embeddings, norms)
            dequantized[name] = weight

    return dequantized


def _load_fp8_weights_without_dequant(
    files: List[str],
    device: torch.device,
    tp_rank: int,
    tp_size: int,
    block_size: Tuple[int, int] = (128, 128),
) -> Dict[str, torch.Tensor]:
    """
    Load FP8 weights without dequantization.

    Returns format:
    - <name>.weight: FP8 tensor (float8_e4m3fn)
    - <name>.weight_scale_inv: scale tensor (bfloat16)

    Critical: Scale merging must match weight merging logic.

    Args:
        files: List of safetensors file paths
        device: Target device
        tp_rank: Tensor parallel rank
        tp_size: Tensor parallel world size
        block_size: FP8 quantization block size

    Returns:
        Dict with FP8 weights and scales
    """
    state_dict: Dict[str, torch.Tensor] = {}
    scale_dict: Dict[str, torch.Tensor] = {}

    device_str = str(device)

    # First pass: load all weights and scales
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device=device_str) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name.endswith("_scale_inv"):
                    scale_dict[name] = tensor
                else:
                    state_dict[name] = tensor

    # Second pass: merge qkv and gate_up weights AND their scales
    state_dict, scale_dict = _merge_fp8_state_dict(state_dict, scale_dict)

    # Validate: each weight must have a corresponding scale with matching shape
    for name, weight in state_dict.items():
        scale_name = f"{name}_scale_inv"
        if scale_name in scale_dict:
            scale = scale_dict[scale_name]
            # Validate scale shape: [O//128, I//128]
            expected_scale_shape = (
                (weight.shape[0] + block_size[0] - 1) // block_size[0],
                (weight.shape[1] + block_size[1] - 1) // block_size[1],
            )
            assert scale.shape == expected_scale_shape, (
                f"Scale shape mismatch for {name}: "
                f"expected {expected_scale_shape}, got {scale.shape}"
            )

    # Third pass: apply TP sharding to FP8 weights and scales
    result: Dict[str, torch.Tensor] = {}

    for name, weight in state_dict.items():
        scale_name = f"{name}_scale_inv"
        if scale_name in scale_dict:
            scale = scale_dict[scale_name]

            # Determine split dimension for TP
            split_dim = 0
            if any(name.endswith(p) for p in [".o_proj", ".down_proj"]):
                split_dim = 1
            elif name.endswith("lm_head") or name.endswith("embed_tokens"):
                split_dim = 0

            # Shard FP8 weight and scale with shared shard bounds
            row_start, row_end, col_start, col_end = _get_fp8_shard_bounds(
                weight.shape, split_dim, tp_rank, tp_size
            )
            result[name] = weight[row_start:row_end, col_start:col_end]
            result[scale_name] = _shard_fp8_scale(
                scale,
                row_start,
                row_end,
                col_start,
                col_end,
                block_size[0],
            )
        else:
            # Non-quantized weights (embeddings, norms)
            result[name] = weight

    return result


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = div_ceil(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_fp8_state_dict(state_dict: Dict[str, torch.Tensor], scale_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Merge FP8 weights (qkv, gate_up) and their scale tensors.
    Must merge scales first, then dequantize the merged weight.
    """
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    filtered_scale_dict: Dict[str, torch.Tensor] = {}

    # Merge qkv_proj - find all layers that have q_proj
    q_keys = [k for k in state_dict if ".q_proj.weight" in k]
    for q_key in q_keys:
        layer_prefix = q_key.replace(".q_proj.weight", "")
        k_key = f"{layer_prefix}.k_proj.weight"
        v_key = f"{layer_prefix}.v_proj.weight"

        if k_key in state_dict and v_key in state_dict:
            q_proj = state_dict[q_key]
            k_proj = state_dict[k_key]
            v_proj = state_dict[v_key]
            new_key = f"{layer_prefix}.qkv_proj.weight"
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)

            # Also merge scales
            q_scale_key = f"{layer_prefix}.q_proj.weight_scale_inv"
            k_scale_key = f"{layer_prefix}.k_proj.weight_scale_inv"
            v_scale_key = f"{layer_prefix}.v_proj.weight_scale_inv"

            if q_scale_key in scale_dict and k_scale_key in scale_dict and v_scale_key in scale_dict:
                q_scale = scale_dict[q_scale_key]
                k_scale = scale_dict[k_scale_key]
                v_scale = scale_dict[v_scale_key]
                # Scales are concatenated along the output dimension (dim 0)
                new_scale_key = f"{layer_prefix}.qkv_proj.weight_scale_inv"
                filtered_scale_dict[new_scale_key] = torch.cat([q_scale, k_scale, v_scale], dim=0)

            # Mark old keys for removal
            for key in [q_key, k_key, v_key]:
                state_dict[key] = None  # Mark as processed
            for key in [q_scale_key, k_scale_key, v_scale_key]:
                if key in scale_dict:
                    scale_dict[key] = None  # Mark as processed

    # Merge gate_up_proj - find all layers that have gate_proj
    gate_keys = [k for k in state_dict if ".gate_proj.weight" in k]
    for gate_key in gate_keys:
        layer_prefix = gate_key.replace(".gate_proj.weight", "")
        up_key = f"{layer_prefix}.up_proj.weight"

        if up_key in state_dict:
            gate_proj = state_dict[gate_key]
            up_proj = state_dict[up_key]
            new_key = f"{layer_prefix}.gate_up_proj.weight"
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)

            # Also merge scales (concatenate along output dim)
            gate_scale_key = f"{layer_prefix}.gate_proj.weight_scale_inv"
            up_scale_key = f"{layer_prefix}.up_proj.weight_scale_inv"

            if gate_scale_key in scale_dict and up_scale_key in scale_dict:
                gate_scale = scale_dict[gate_scale_key]
                up_scale = scale_dict[up_scale_key]
                # Scales: [out//128, in//128] - concat along dim 0 (output)
                new_scale_key = f"{layer_prefix}.gate_up_proj.weight_scale_inv"
                filtered_scale_dict[new_scale_key] = torch.cat([gate_scale, up_scale], dim=0)

            # Mark old keys for removal
            for key in [gate_key, up_key]:
                state_dict[key] = None  # Mark as processed
            for key in [gate_scale_key, up_scale_key]:
                if key in scale_dict:
                    scale_dict[key] = None  # Mark as processed

    # Add remaining weights and scales (skip processed ones)
    for key, value in state_dict.items():
        if value is not None:
            filtered_state_dict[key] = value
    for key, value in scale_dict.items():
        if value is not None:
            filtered_scale_dict[key] = value

    return filtered_state_dict, filtered_scale_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_weight(
    model_path: str,
    device: torch.device,
    fp8_keep_quantized: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Load model weights from HuggingFace format.

    Args:
        model_path: HuggingFace model ID or local path
        device: Target device for weights
        fp8_keep_quantized: If True, keep FP8 weights in quantized format
            instead of dequantizing to BF16. This saves memory but requires
            on-the-fly dequantization during forward pass.

    Returns:
        Dict mapping parameter names to tensors
    """
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")

    # Get TP info, default to single GPU (rank=0, size=1) if not set
    tp_info = try_get_tp_info()
    if tp_info is None:
        tp_rank = 0
        tp_size = 1
    else:
        tp_rank = tp_info.rank
        tp_size = tp_info.size
    disable_tqdm = (tp_rank != 0) if tp_size > 1 else False

    # Check if this is a quantized model
    is_fp8 = _is_fp8_model(model_folder)
    is_awq = _is_awq_model(model_folder)

    if is_fp8:
        # Load config to get block_size
        config_path = os.path.join(model_folder, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        quant_config = config.get("quantization_config", {})
        block_size = tuple(quant_config.get("weight_block_size", [128, 128]))

        if fp8_keep_quantized:
            # Keep FP8 weights in quantized format for memory efficiency
            state_dict = _load_fp8_weights_without_dequant(
                files, device, tp_rank, tp_size, block_size
            )
        else:
            # Load and dequantize FP8 weights (includes merging qkv/gate_up)
            state_dict = _load_fp8_weights(files, device, tp_rank, tp_size, block_size)
        # FP8 already applied TP sharding during loading
    elif is_awq:
        # TODO: AWQ support - requires different dequantization logic
        raise NotImplementedError("AWQ quantization is not yet supported")
    else:
        # Load non-quantized weights (BF16/FP16)
        state_dict: Dict[str, torch.Tensor] = {}
        device_str = str(device)

        for file in tqdm(sorted(files), desc="Loading weights", disable=disable_tqdm):
            with safetensors.safe_open(file, framework="pt", device=device_str) as f:
                for name in f.keys():
                    state_dict[name] = f.get_tensor(name)

        # Apply tensor parallelism sharding
        if tp_size > 1:
            state_dict = _shard_state_dict(state_dict)

        # Merge qkv and gate_up weights
        state_dict = _merge_state_dict(state_dict)

    return state_dict
