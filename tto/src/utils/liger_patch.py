"""Opt-in, instance-local Liger integration for the supported language models."""

from dataclasses import dataclass, fields
from types import MethodType

from torch import nn


def _replicate_liger_module(self):
    replica = type(self)._replicate_for_data_parallel(self)
    # DataParallel shallow-copies __dict__; otherwise forward stays bound to
    # the original module and uses its parameters on the wrong GPU.
    replica.forward = MethodType(self.forward.__func__, replica)
    replica._replicate_for_data_parallel = MethodType(_replicate_liger_module, replica)
    return replica


def _patch_forward(module, forward):
    module.forward = MethodType(forward, module)
    module._replicate_for_data_parallel = MethodType(_replicate_liger_module, module)


@dataclass(frozen=True)
class LigerKernelConfig:
    """Select kernels explicitly. Omitted components keep their native implementation."""

    rope: bool = False
    rms_norm: bool = False
    layer_norm: bool = False
    swiglu: bool = False
    geglu: bool = False
    cross_entropy: bool = False
    fused_linear_cross_entropy: bool = False

    def __post_init__(self):
        for field in fields(self):
            if not isinstance(getattr(self, field.name), bool):
                raise TypeError(f"liger_kernel.{field.name} must be a boolean.")
        if self.cross_entropy and self.fused_linear_cross_entropy:
            raise ValueError("Choose cross_entropy or fused_linear_cross_entropy, not both.")
        if self.swiglu and self.geglu:
            raise ValueError("Choose swiglu or geglu, not both.")

    @property
    def enabled(self) -> bool:
        return any(getattr(self, field.name) for field in fields(self))


def patch_model_with_liger(model, config: LigerKernelConfig):
    """Patch only this model's instances, preserving parameters and checkpoint keys.

    Kernels execute on supported accelerators. Patching itself can happen before
    moving the model to an accelerator. Save/load retains ordinary model weights;
    reapply this runtime configuration when resuming training.
    """
    if not isinstance(config, LigerKernelConfig):
        raise TypeError("config must be a LigerKernelConfig.")

    from ..models.titans.modeling_titans import (
        TitansAttention,
        TitansForCausalLM,
        TitansMLP,
        TitansRMSNorm,
    )
    from ..models.ttt.modeling_ttt import (
        TTTBaseMemory,
        TTTForCausalLM,
        TTTGatedMLP,
        TTTRMSNorm,
    )

    if not isinstance(model, (TitansForCausalLM, TTTForCausalLM)):
        raise TypeError("Liger integration supports TitansForCausalLM and TTTForCausalLM.")
    if config.layer_norm and not isinstance(model, TTTForCausalLM):
        raise ValueError("layer_norm is supported only by TTT.")
    if config.swiglu and model.config.hidden_act not in {"silu", "swish"}:
        raise ValueError("swiglu requires hidden_act='silu' or 'swish'.")
    if config.geglu and model.config.hidden_act not in {"gelu_pytorch_tanh", "gelu_new"}:
        raise ValueError("geglu requires a tanh-approximate GELU activation, not exact GELU.")
    if config.rope and isinstance(model, TitansForCausalLM) and model.config.variant == "lmm":
        raise ValueError("Titans LMM has no RoPE. Enable rope only for attention variants.")

    previous = getattr(model, "_liger_kernel_config", None)
    if previous is not None:
        if previous != config:
            raise ValueError("This model is already patched with a different Liger configuration.")
        return model
    if not config.enabled:
        return model

    # Import the optional dependency only when at least one kernel is requested.
    from .liger_ops import (
        geglu_forward,
        layer_norm_forward,
        liger_causal_lm_forward,
        liger_causal_loss,
        liger_rope,
        rms_norm_forward,
        swiglu_forward,
    )

    for module in model.modules():
        if config.rms_norm and isinstance(module, (TitansRMSNorm, TTTRMSNorm)):
            _patch_forward(module, rms_norm_forward)
        if config.layer_norm and isinstance(module, nn.LayerNorm):
            _patch_forward(module, layer_norm_forward)
        if isinstance(module, (TitansMLP, TTTGatedMLP)):
            if config.swiglu:
                _patch_forward(module, swiglu_forward)
            elif config.geglu:
                _patch_forward(module, geglu_forward)
        if config.rope and isinstance(module, (TitansAttention, TTTBaseMemory)):
            module.apply_rotary_pos_emb = liger_rope

    if config.cross_entropy:
        model.loss_function = liger_causal_loss
    if config.fused_linear_cross_entropy:
        _patch_forward(model, liger_causal_lm_forward)
    model._liger_kernel_config = config
    return model