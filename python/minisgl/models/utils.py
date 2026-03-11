from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    Fp8LinearColParallelMerged,
    Fp8LinearColParallelMergedV2,
    Fp8LinearOProj,
    Fp8LinearQKVMerged,
    Fp8LinearRowParallel,
    Fp8LinearRowParallelV2,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseOP):
    """Gated MLP with optional FP8 support.

    Args:
        config: Model configuration
        use_fp8: If True, use FP8 weights with weight dequantization
        use_fp8_input_quant: If True, use FP8 input quantization (requires use_fp8=True)
    """

    def __init__(
        self,
        config: ModelConfig,
        use_fp8: bool = False,
        use_fp8_input_quant: bool = False,
        fp8_input_scale_method: str = "per_tensor",
    ):
        self.use_fp8_input_quant = use_fp8_input_quant

        if use_fp8_input_quant:
            # New FP8 Input Quantization mode (V2 layers)
            self.gate_up_proj = Fp8LinearColParallelMergedV2(
                config.hidden_size,
                [config.intermediate_size, config.intermediate_size],
                has_bias=False,
                input_scale_method=fp8_input_scale_method,
            )
        elif use_fp8:
            # Existing FP8 weight dequantization mode
            self.gate_up_proj = Fp8LinearColParallelMerged(
                config.hidden_size,
                [config.intermediate_size, config.intermediate_size],
                has_bias=False,
            )
        else:
            # BF16 mode
            self.gate_up_proj = LinearColParallelMerged(
                config.hidden_size,
                [config.intermediate_size, config.intermediate_size],
                has_bias=False,
            )

        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn

        if use_fp8_input_quant:
            # New FP8 Input Quantization mode (V2 layers)
            self.down_proj = Fp8LinearRowParallelV2(
                config.intermediate_size,
                config.hidden_size,
                has_bias=False,
                input_scale_method=fp8_input_scale_method,
            )
        elif use_fp8:
            # Existing FP8 weight dequantization mode
            self.down_proj = Fp8LinearRowParallel(
                config.intermediate_size,
                config.hidden_size,
                has_bias=False,
            )
        else:
            # BF16 mode
            self.down_proj = LinearRowParallel(
                config.intermediate_size,
                config.hidden_size,
                has_bias=False,
            )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)

        return final_hidden_states


class RopeAttn(BaseOP):
    """Attention with RoPE.

    Note: QKV projection stays in BF16 for RoPE compatibility.
    FP8 input quantization is only applied to MLP layers.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
        use_fp8: bool = False,
        use_fp8_input_quant: bool = False,  # Not used for attention, kept for API consistency
    ):
        head_dim = config.head_dim
        # QKV projection: use FP8 weight dequantization if use_fp8, but NOT input quantization
        # This is because RoPE requires BF16 precision
        if use_fp8:
            self.qkv_proj = Fp8LinearQKVMerged(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_qo_heads=config.num_qo_heads,
                num_kv_heads=config.num_kv_heads,
                has_bias=has_attn_bias,
            )
        else:
            self.qkv_proj = LinearQKVMerged(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_qo_heads=config.num_qo_heads,
                num_kv_heads=config.num_kv_heads,
                has_bias=has_attn_bias,
            )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        # O projection: use FP8 weight dequantization if use_fp8
        if use_fp8:
            self.o_proj = Fp8LinearOProj(
                head_dim * config.num_qo_heads,
                config.hidden_size,
                has_bias=False,
            )
        else:
            self.o_proj = LinearOProj(
                head_dim * config.num_qo_heads,
                config.hidden_size,
                has_bias=False,
            )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
