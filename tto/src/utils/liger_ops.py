"""Instance forwards backed by Liger's public functional interfaces."""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from liger_kernel.transformers.functional import (
    liger_cross_entropy,
    liger_geglu,
    liger_layer_norm,
    liger_rms_norm,
    liger_rope,
    liger_swiglu,
)
from liger_kernel.transformers.model.loss_utils import (
    LigerForCausalLMLoss,
    unpack_cross_entropy_result,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import slice_logits_to_keep
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def rms_norm_forward(self, hidden_states):
    # Residual branches can share incoming gradients. Do not overwrite them.
    return liger_rms_norm(
        hidden_states, self.weight, self.variance_epsilon, casting_mode="llama", in_place=False
    )


def layer_norm_forward(self, hidden_states):
    return liger_layer_norm(hidden_states, self.weight, self.bias, self.eps)


def _gated_mlp_forward(self, hidden_states, activation):
    tp = self.config.pretraining_tp if self.config.model_type == "ttt" else 1
    if tp == 1:
        return self.down_proj(activation(self.gate_proj(hidden_states), self.up_proj(hidden_states)))

    # Preserve TTT's existing tensor-parallel projection/reduction order.
    width = self.intermediate_size // tp
    gate = torch.cat([F.linear(hidden_states, w) for w in self.gate_proj.weight.split(width, dim=0)], dim=-1)
    up = torch.cat([F.linear(hidden_states, w) for w in self.up_proj.weight.split(width, dim=0)], dim=-1)
    intermediate = activation(gate, up).split(width, dim=-1)
    return sum(F.linear(x, w) for x, w in zip(intermediate, self.down_proj.weight.split(width, dim=1)))


def swiglu_forward(self, hidden_states):
    return _gated_mlp_forward(self, hidden_states, liger_swiglu)


def geglu_forward(self, hidden_states):
    return _gated_mlp_forward(self, hidden_states, liger_geglu)


def liger_causal_loss(
    logits,
    labels,
    vocab_size,
    num_items_in_batch=None,
    ignore_index=-100,
    shift_labels=None,
    **kwargs,
):
    """Liger CE with the Transformers causal-target and token-normalization contract."""
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    # Liger CE reuses its input for gradients. Keep returned FP32 logits intact;
    # lower-precision logits already get a separate allocation from the upcast.
    logits = logits.to(
        dtype=torch.float32, copy=logits.requires_grad and logits.dtype == torch.float32
    ).reshape(-1, vocab_size)
    shift_labels = shift_labels.reshape(-1).to(logits.device)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = liger_cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction=reduction)
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


def liger_causal_lm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_params=None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    """Fuse the LM head and loss without copying either model's backbone."""
    is_ttt = self.config.model_type == "ttt"
    if is_ttt and output_attentions:
        raise ValueError("output_attentions is not available in TTTForCausalLM.")
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    shift_labels = kwargs.pop("shift_labels", None)
    has_targets = labels is not None or shift_labels is not None
    if skip_logits is None:
        skip_logits = self.training and has_targets
    if skip_logits and not has_targets:
        raise ValueError("skip_logits=True requires labels or shift_labels.")

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_params=cache_params,
        **kwargs,
    )
    hidden_states = slice_logits_to_keep(outputs[0], logits_to_keep)
    loss = None
    logits = None
    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=hidden_states.contiguous(),
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss = unpack_cross_entropy_result(result)[0]
    else:
        if is_ttt and self.config.pretraining_tp > 1:
            width = self.vocab_size // self.config.pretraining_tp
            logits = torch.cat([F.linear(hidden_states, w) for w in self.lm_head.weight.split(width)], dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        if is_ttt:
            logits = logits.float()
        if has_targets:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        result = (logits,) + outputs.to_tuple()[1:]
        return (loss,) + result if loss is not None else result
    if is_ttt:
        from ..models.ttt.modeling_ttt import TTTCausalLMOutput

        return TTTCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
