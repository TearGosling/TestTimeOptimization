from __future__ import annotations

import math
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.masking_utils import flash_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, ModelOutput, logging
from transformers.utils.generic import is_flash_attention_requested

try:
    from .configuration_titans import TitansConfig
except ImportError:  # pragma: no cover - supports direct file imports.
    from configuration_titans import TitansConfig


logger = logging.get_logger(__name__)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, k_rot = q[..., :rotary_dim], k[..., :rotary_dim]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    if rotary_dim != q.shape[-1]:
        q_embed = torch.cat([q_embed, q[..., rotary_dim:]], dim=-1)
        k_embed = torch.cat([k_embed, k[..., rotary_dim:]], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query.float(), key.transpose(-1, -2).float()) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(value.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


def _activation_grad(name: str, x: torch.Tensor) -> torch.Tensor:
    name = name.lower()
    if name in {"silu", "swish"}:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))
    if name == "gelu":
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) + x * torch.exp(-0.5 * x * x) / math.sqrt(
            2.0 * math.pi
        )
    if name == "relu":
        return (x > 0).to(dtype=x.dtype)
    if name == "tanh":
        y = torch.tanh(x)
        return 1.0 - y * y
    raise ValueError(f"Unsupported differentiable memory activation {name!r}.")


class TitansRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TitansRotaryEmbedding(nn.Module):
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = config.rope_parameters["rope_type"]
        rope_init_fn = self.compute_default_rope_parameters if self.rope_type == "default" else ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config: TitansConfig, **kwargs) -> tuple[torch.Tensor, float]:
        dim = config.hidden_size // config.num_attention_heads
        dim = int(dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
        base = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float32)
        emb = torch.cat([freqs, freqs], dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


class TitansMLP(nn.Module):
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TitansDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.state_size = max(kernel_size - 1, 0)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional["TitansCache"] = None,
        layer_idx: Optional[int] = None,
        namespace: str = "memory",
        name: str = "q",
        use_cache: bool = False,
    ) -> torch.Tensor:
        if self.kernel_size == 1:
            return hidden_states

        seq_len = hidden_states.shape[1]
        conv_input = hidden_states.transpose(1, 2)
        if past_key_values is not None and use_cache:
            state = past_key_values.get_conv_state(
                namespace,
                name,
                layer_idx,
                conv_input.shape[0],
                conv_input.shape[1],
                self.state_size,
                conv_input.device,
                conv_input.dtype,
            )
            padded = torch.cat([state, conv_input], dim=-1)
            out = F.conv1d(padded, self.conv.weight, self.conv.bias, padding=0, groups=self.channels)
            past_key_values.set_conv_state(namespace, name, layer_idx, padded[..., -self.state_size :].detach().clone())
        else:
            out = self.conv(conv_input)[..., :seq_len]
        return out.transpose(1, 2)


class TitansCache:
    """
    Cache for Titans inference.

    It stores the adaptive long-term memory weights and surprise momentum,
    Q/K/V convolution states, and the short-term attention KV cache.  For the
    attention-free `"lmm"` variant the attention key/value lists are empty to
    avoid allocating memory for unused state.
    """

    is_compileable = False

    def __init__(
        self,
        config: TitansConfig,
        max_batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        model: Optional["TitansModel"] = None,
    ):
        self.config = config
        self.seqlen_offset = 0
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.batch_caches: Optional[list[Optional["TitansCache"]]] = None
        self.memory_weights: list[Optional[list[torch.Tensor]]] = [None for _ in range(config.num_hidden_layers)]
        self.memory_biases: list[Optional[list[torch.Tensor]]] = [None for _ in range(config.num_hidden_layers)]
        self.memory_surprise_weights: list[Optional[list[torch.Tensor]]] = [
            None for _ in range(config.num_hidden_layers)
        ]
        self.memory_surprise_biases: list[Optional[list[torch.Tensor]]] = [
            None for _ in range(config.num_hidden_layers)
        ]
        self.memory_chunk_states: list[Optional[dict[str, list[torch.Tensor]]]] = [
            None for _ in range(config.num_hidden_layers)
        ]
        self.memory_chunk_inputs: list[Optional[tuple[torch.Tensor, ...]]] = [
            None for _ in range(config.num_hidden_layers)
        ]
        self.mac_segment_states: list[Optional[dict[str, list[torch.Tensor]]]] = [
            None for _ in range(config.num_hidden_layers)
        ]
        self.mac_segment_inputs: list[Optional[tuple[torch.Tensor, ...]]] = [
            None for _ in range(config.num_hidden_layers)
        ]

        if config.variant == "lmm":
            self.key_cache: list[Optional[torch.Tensor]] = []
            self.value_cache: list[Optional[torch.Tensor]] = []
        else:
            self.key_cache = [None for _ in range(config.num_hidden_layers)]
            self.value_cache = [None for _ in range(config.num_hidden_layers)]

        self.memory_conv_states = {name: [None for _ in range(config.num_hidden_layers)] for name in ("q", "k", "v")}
        self.attention_conv_states = (
            {}
            if config.variant == "lmm"
            else {name: [None for _ in range(config.num_hidden_layers)] for name in ("q", "k", "v")}
        )

        if model is not None:
            for layer_idx, layer in enumerate(model.layers):
                state = layer.memory.initial_state(max_batch_size, device=device, dtype=dtype)
                self.set_memory_state(layer_idx, state, detach=True)

    def __len__(self) -> int:
        return self.config.num_hidden_layers

    def get_memory_state(self, layer_idx: int) -> Optional[dict[str, list[torch.Tensor]]]:
        weights = self.memory_weights[layer_idx]
        if weights is None:
            return None
        return {
            "weights": weights,
            "biases": self.memory_biases[layer_idx],
            "surprise_weights": self.memory_surprise_weights[layer_idx],
            "surprise_biases": self.memory_surprise_biases[layer_idx],
        }

    def set_memory_state(self, layer_idx: int, state: dict[str, list[torch.Tensor]], detach: bool = True) -> None:
        def maybe_detach(values: list[torch.Tensor]) -> list[torch.Tensor]:
            return [value.detach() if detach else value for value in values]

        self.memory_weights[layer_idx] = maybe_detach(state["weights"])
        self.memory_biases[layer_idx] = maybe_detach(state["biases"])
        self.memory_surprise_weights[layer_idx] = maybe_detach(state["surprise_weights"])
        self.memory_surprise_biases[layer_idx] = maybe_detach(state["surprise_biases"])

    def get_conv_state(
        self,
        namespace: str,
        name: str,
        layer_idx: int,
        batch_size: int,
        channels: int,
        state_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state_size == 0:
            return torch.empty(batch_size, channels, 0, device=device, dtype=dtype)
        container = self.memory_conv_states if namespace == "memory" else self.attention_conv_states
        state = container[name][layer_idx]
        if (
            state is None
            or state.shape[0] != batch_size
            or state.shape[1] != channels
            or state.shape[2] != state_size
            or state.device != device
            or state.dtype != dtype
        ):
            state = torch.zeros(batch_size, channels, state_size, device=device, dtype=dtype)
            container[name][layer_idx] = state
        return state

    def set_conv_state(self, namespace: str, name: str, layer_idx: int, state: torch.Tensor) -> None:
        container = self.memory_conv_states if namespace == "memory" else self.attention_conv_states
        container[name][layer_idx] = state

    def update_attention(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        sliding_window: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.key_cache:
            return key_states, value_states
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states.detach()
            self.value_cache[layer_idx] = value_states.detach()
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states.detach()], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states.detach()], dim=2)
        key_states, value_states = self.key_cache[layer_idx], self.value_cache[layer_idx]
        if sliding_window is not None:
            self.key_cache[layer_idx] = key_states[:, :, -sliding_window:, :].clone()
            self.value_cache[layer_idx] = value_states[:, :, -sliding_window:, :].clone()
        # Early queries still need their own window, not just the last query's.
        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.seqlen_offset

    def _select_batch(self, indices: torch.LongTensor) -> "TitansCache":
        selected = copy(self)
        for name in (
            "memory_weights", "memory_biases", "memory_surprise_weights", "memory_surprise_biases",
            "memory_chunk_states", "memory_chunk_inputs", "mac_segment_states", "mac_segment_inputs",
            "key_cache", "value_cache",
        ):
            setattr(selected, name, getattr(self, name).copy())
        selected.memory_conv_states = {name: states.copy() for name, states in self.memory_conv_states.items()}
        selected.attention_conv_states = {name: states.copy() for name, states in self.attention_conv_states.items()}
        selected.reorder_cache(indices)
        return selected

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.max_batch_size = beam_idx.numel()
        if self.batch_caches is not None:
            self.batch_caches = [
                self.batch_caches[index]._select_batch(beam_idx.new_zeros(1))
                if self.batch_caches[index] is not None else None
                for index in beam_idx.tolist()
            ]
            return
        def reorder_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            return value.index_select(0, beam_idx.to(value.device))

        for layer_idx in range(self.config.num_hidden_layers):
            if self.key_cache:
                self.key_cache[layer_idx] = reorder_tensor(self.key_cache[layer_idx])
                self.value_cache[layer_idx] = reorder_tensor(self.value_cache[layer_idx])
            for container in (self.memory_conv_states, self.attention_conv_states):
                for states in container.values():
                    states[layer_idx] = reorder_tensor(states[layer_idx])
            state = self.get_memory_state(layer_idx)
            if state is not None:
                reordered = {key: [reorder_tensor(value) for value in values] for key, values in state.items()}
                self.set_memory_state(layer_idx, reordered, detach=True)
            for states in (self.memory_chunk_states, self.mac_segment_states):
                state = states[layer_idx]
                if state is not None:
                    states[layer_idx] = {key: [reorder_tensor(value) for value in values] for key, values in state.items()}
            for inputs in (self.memory_chunk_inputs, self.mac_segment_inputs):
                if inputs[layer_idx] is not None:
                    inputs[layer_idx] = tuple(reorder_tensor(value) for value in inputs[layer_idx])


class TitansNeuralMemory(nn.Module):
    def __init__(self, config: TitansConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.memory_num_heads
        self.head_dim = config.memory_head_dim
        self.memory_width = self.num_heads * self.head_dim
        self.chunk_size = config.memory_chunk_size
        self.act_name = config.memory_activation
        self.act_fn = ACT2FN[self.act_name]

        self.q_proj = nn.Linear(config.hidden_size, self.memory_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.memory_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.memory_width, bias=False)
        self.control_proj = nn.Linear(config.hidden_size, self.num_heads * 3, bias=True)
        self.out_norm = TitansRMSNorm(self.memory_width, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, self.memory_width, bias=True)
        self.out_proj = nn.Linear(self.memory_width, config.hidden_size, bias=False)

        if config.use_memory_convolution:
            self.q_conv = TitansDepthwiseConv1d(self.memory_width, config.memory_qkv_conv_kernel)
            self.k_conv = TitansDepthwiseConv1d(self.memory_width, config.memory_qkv_conv_kernel)
            self.v_conv = TitansDepthwiseConv1d(self.memory_width, config.memory_qkv_conv_kernel)
        else:
            self.q_conv = self.k_conv = self.v_conv = None

        dims = [self.head_dim]
        if config.memory_num_layers > 1:
            dims.extend([config.memory_hidden_size for _ in range(config.memory_num_layers - 1)])
        dims.append(self.head_dim)
        self.memory_dims = dims

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            # Used to be torch.empty for self.weights but there's some sort of weird initialization bug
            # where the weights never actually get filled. Hacky fix until I figure it out.
            self.weights.append(nn.Parameter(torch.randn(self.num_heads, in_dim, out_dim) * self.config.initializer_range))
            self.biases.append(nn.Parameter(torch.zeros(self.num_heads, 1, out_dim)))

    def initial_state(
        self,
        batch_size: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, list[torch.Tensor]]:
        dtype = self.weights[0].dtype if dtype is None else dtype
        weights = [weight.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).clone() for weight in self.weights]
        biases = [bias.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).clone() for bias in self.biases]
        return {
            "weights": weights,
            "biases": biases,
            "surprise_weights": [torch.zeros_like(weight) for weight in weights],
            "surprise_biases": [torch.zeros_like(bias) for bias in biases],
        }

    def _state_from_cache_or_init(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[TitansCache],
    ) -> dict[str, list[torch.Tensor]]:
        if past_key_values is not None:
            cached = past_key_values.get_memory_state(self.layer_idx)
            if cached is not None:
                return cached
        return self.initial_state(hidden_states.shape[0], hidden_states.device, hidden_states.dtype)

    def _project(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[TitansCache],
        use_cache: bool,
        project_query: bool = True,
        project_update: bool = True,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states) if project_query else None
        k = self.k_proj(hidden_states) if project_update else None
        v = self.v_proj(hidden_states) if project_update else None

        if self.q_conv is not None and project_query:
            q = self.q_conv(q, past_key_values, self.layer_idx, "memory", "q", use_cache)
        if self.k_conv is not None and project_update:
            k = self.k_conv(k, past_key_values, self.layer_idx, "memory", "k", use_cache)
            v = self.v_conv(v, past_key_values, self.layer_idx, "memory", "v", use_cache)

        if q is not None:
            q = F.silu(q).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)
            q = l2norm(q, dim=-1, eps=self.config.l2_norm_eps).transpose(1, 2)
        if k is not None:
            k = F.silu(k).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)
            k = l2norm(k, dim=-1, eps=self.config.l2_norm_eps).transpose(1, 2)
        if v is not None:
            v = F.silu(v).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)
            v = v.transpose(1, 2)

        controls = self.control_proj(hidden_states).view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_heads, 3
        )
        alpha = torch.sigmoid(controls[..., 0]) * self.config.memory_alpha_scale
        eta = torch.sigmoid(controls[..., 1]) * self.config.memory_eta_scale
        theta = torch.sigmoid(controls[..., 2]) * self.config.memory_theta_scale
        alpha = alpha.clamp(0.0, 1.0).transpose(1, 2)
        eta = eta.clamp(0.0, 1.0).transpose(1, 2)
        theta = (theta / math.sqrt(self.head_dim)).transpose(1, 2)
        return q, k, v, theta, eta, alpha

    def _mlp_forward(
        self,
        inputs: torch.Tensor,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        acts = [inputs]
        preacts = []
        hidden = inputs
        for idx, (weight, bias) in enumerate(zip(weights, biases)):
            hidden = torch.einsum("bhki,bhio->bhko", hidden, weight) + bias
            preacts.append(hidden)
            if idx != len(weights) - 1:
                hidden = self.act_fn(hidden)
            acts.append(hidden)
        if self.config.memory_residual:
            hidden = inputs + hidden
            
        return hidden, acts, preacts

    def _gradient_factors(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        predictions, acts, preacts = self._mlp_forward(keys, weights, biases)
        delta = (predictions - values) * self.config.memory_loss_scale
        deltas: list[torch.Tensor] = [None for _ in weights]
        for idx in reversed(range(len(weights))):
            deltas[idx] = delta
            if idx > 0:
                delta = torch.einsum("bhko,bhio->bhki", delta, weights[idx])
                delta = delta * _activation_grad(self.act_name, preacts[idx - 1])
        return acts, deltas

    def _build_chunk_operators(
        self,
        theta: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scalar coefficients for the chunk-start-gradient momentum/decay recurrence."""
        size = theta.shape[-1]
        positions = torch.arange(size, device=theta.device)
        below_diagonal = positions[:, None] > positions[None, :]
        dtype = torch.float64 if theta.dtype == torch.float64 else torch.float32
        # Accumulate controls outside autocast. Interval products handle exact
        # zeros and underflow without logarithms or division by prefix products.
        with torch.autocast(device_type=theta.device.type, enabled=False):
            eta = eta.to(dtype)
            gamma = 1.0 - alpha.to(dtype)
            eta_prod = eta.cumprod(dim=-1)
            gamma_prod = gamma.cumprod(dim=-1)
            eta_intervals = torch.where(below_diagonal, eta.unsqueeze(-1), 1.0).cumprod(dim=-2).tril()
            gamma_intervals = torch.where(below_diagonal, gamma.unsqueeze(-1), 1.0).cumprod(dim=-2).tril()
            sigma = (gamma_intervals @ eta_prod.unsqueeze(-1)).squeeze(-1)
            surprise_coeff = eta_intervals * theta.to(dtype).unsqueeze(-2)
            weight_coeff = gamma_intervals @ surprise_coeff
        return tuple(
            coefficient.to(theta.dtype)
            for coefficient in (eta_prod, gamma_prod, sigma, surprise_coeff, weight_coeff)
        )

    def _parallel_chunk(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        state: dict[str, list[torch.Tensor]],
        update: bool,
        output_after_update: bool,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        if not update:
            output = self._mlp_forward(queries, state["weights"], state["biases"])[0]
            return output, state

        acts, deltas = self._gradient_factors(keys, values, state["weights"], state["biases"])
        eta_prod, gamma_prod, sigma, surprise_coeff, weight_coeff = self._build_chunk_operators(theta, eta, alpha)
        surprise_last = surprise_coeff[:, :, -1, :, None]
        weight_last = weight_coeff[:, :, -1, :, None]
        eta_last = eta_prod[:, :, -1, None, None]
        gamma_last = gamma_prod[:, :, -1, None, None]
        sigma_last = sigma[:, :, -1, None, None]
        next_weights, next_biases = [], []
        next_surprise_weights, next_surprise_biases = [], []

        for idx, delta in enumerate(deltas):
            # Each per-token gradient is acts[idx][t] outer delta[t].
            # Contract tokens before creating a weight-sized tensor.
            surprise_delta = surprise_last * delta
            weight_delta = weight_last * delta
            next_surprise_weights.append(
                eta_last * state["surprise_weights"][idx]
                - torch.einsum("bhki,bhko->bhio", acts[idx], surprise_delta)
            )
            next_weights.append(
                gamma_last * state["weights"][idx] + sigma_last * state["surprise_weights"][idx]
                - torch.einsum("bhki,bhko->bhio", acts[idx], weight_delta)
            )
            next_surprise_biases.append(
                eta_last * state["surprise_biases"][idx] - surprise_delta.sum(dim=2, keepdim=True)
            )
            next_biases.append(
                gamma_last * state["biases"][idx] + sigma_last * state["surprise_biases"][idx]
                - weight_delta.sum(dim=2, keepdim=True)
            )

        if output_after_update:
            hidden = queries
            gamma_col = gamma_prod.unsqueeze(-1)
            sigma_col = sigma.unsqueeze(-1)
            for idx, delta in enumerate(deltas):
                base = torch.einsum("bhki,bhio->bhko", hidden, state["weights"][idx]) + state["biases"][idx]
                momentum = (
                    torch.einsum("bhki,bhio->bhko", hidden, state["surprise_weights"][idx])
                    + state["surprise_biases"][idx]
                )
                # The +1 folds the bias gradient into the same dual contraction.
                similarities = torch.einsum("bhki,bhji->bhkj", hidden, acts[idx]) + 1.0
                hidden = gamma_col * base + sigma_col * momentum - (weight_coeff * similarities) @ delta
                if idx != len(deltas) - 1:
                    hidden = self.act_fn(hidden)
            output = queries + hidden if self.config.memory_residual else hidden
        else:
            output = self._mlp_forward(queries, state["weights"], state["biases"])[0]
        next_state = {
            "weights": next_weights,
            "biases": next_biases,
            "surprise_weights": next_surprise_weights,
            "surprise_biases": next_surprise_biases,
        }
        return output, next_state

    def _sequential_chunk(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        state: dict[str, list[torch.Tensor]],
        update: bool,
        output_after_update: bool,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        outputs = []
        cur_state = state
        for token_idx in range(queries.shape[2]):
            q = queries[:, :, token_idx : token_idx + 1]
            if update:
                acts, deltas = self._gradient_factors(
                    keys[:, :, token_idx : token_idx + 1],
                    values[:, :, token_idx : token_idx + 1],
                    cur_state["weights"],
                    cur_state["biases"],
                )
                next_state = {"weights": [], "biases": [], "surprise_weights": [], "surprise_biases": []}
                for idx in range(len(cur_state["weights"])):
                    s_w = eta[:, :, token_idx].view(*eta.shape[:2], 1, 1) * cur_state["surprise_weights"][idx]
                    grad_w = torch.einsum("bhki,bhko->bhio", acts[idx], deltas[idx])
                    s_w = s_w - theta[:, :, token_idx].view(*theta.shape[:2], 1, 1) * grad_w
                    w = (1.0 - alpha[:, :, token_idx]).view(*alpha.shape[:2], 1, 1) * cur_state["weights"][idx] + s_w
                    s_b = eta[:, :, token_idx].view(*eta.shape[:2], 1, 1) * cur_state["surprise_biases"][idx]
                    s_b = s_b - theta[:, :, token_idx].view(*theta.shape[:2], 1, 1) * deltas[idx]
                    b = (1.0 - alpha[:, :, token_idx]).view(*alpha.shape[:2], 1, 1) * cur_state["biases"][idx] + s_b
                    next_state["weights"].append(w)
                    next_state["biases"].append(b)
                    next_state["surprise_weights"].append(s_w)
                    next_state["surprise_biases"].append(s_b)
                if output_after_update:
                    cur_state = next_state
                output = self._mlp_forward(q, cur_state["weights"], cur_state["biases"])[0]
                cur_state = next_state
            else:
                output = self._mlp_forward(q, cur_state["weights"], cur_state["biases"])[0]
            outputs.append(output)
        return torch.cat(outputs, dim=2), cur_state

    def _output_projection(self, memory_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        memory_output = self.out_norm(memory_output) * F.silu(self.gate_proj(hidden_states))
        return self.out_proj(memory_output)

    def retrieve(
        self,
        hidden_states: torch.Tensor,
        state: dict[str, list[torch.Tensor]],
        past_key_values: Optional[TitansCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        queries, _, _, _, _, _ = self._project(
            hidden_states, past_key_values, use_cache, project_query=True, project_update=False
        )
        memory_output = self._mlp_forward(queries, state["weights"], state["biases"])[0]
        memory_output = memory_output.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[1], self.memory_width)
        return self._output_projection(memory_output, hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[TitansCache] = None,
        use_cache: bool = False,
        state: Optional[dict[str, list[torch.Tensor]]] = None,
        update: bool = True,
        output_after_update: bool = True,
        update_cache: bool = True,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        state = self._state_from_cache_or_init(hidden_states, past_key_values) if state is None else state
        queries, keys, values, theta, eta, alpha = self._project(hidden_states, past_key_values, use_cache)
        cache_chunk = (
            past_key_values is not None and use_cache and update_cache and update
            and self.config.use_parallel_memory_training
        )
        projected = (queries, keys, values, theta, eta, alpha)
        pending_length = 0
        if cache_chunk:
            pending = past_key_values.memory_chunk_inputs[self.layer_idx]
            if pending is not None:
                pending_length = pending[0].shape[2]
                projected = tuple(torch.cat([old, new], dim=2) for old, new in zip(pending, projected))
                state = past_key_values.memory_chunk_states[self.layer_idx]
        queries, keys, values, theta, eta, alpha = projected

        outputs = []
        cur_state = state
        scan_fn = self._parallel_chunk if self.config.use_parallel_memory_training else self._sequential_chunk
        for start in range(0, queries.shape[2], self.chunk_size):
            end = min(start + self.chunk_size, queries.shape[2])
            chunk_start_state = cur_state
            chunk_output, cur_state = scan_fn(
                queries[:, :, start:end],
                keys[:, :, start:end],
                values[:, :, start:end],
                theta[:, :, start:end],
                eta[:, :, start:end],
                alpha[:, :, start:end],
                cur_state,
                update=update,
                output_after_update=output_after_update,
            )
            outputs.append(chunk_output)

        if cache_chunk:
            remainder = queries.shape[2] % self.chunk_size
            past_key_values.memory_chunk_inputs[self.layer_idx] = (
                tuple(value[:, :, -remainder:].detach().clone() for value in projected) if remainder else None
            )
            past_key_values.memory_chunk_states[self.layer_idx] = (
                {key: [value.detach() for value in values] for key, values in chunk_start_state.items()}
                if remainder else None
            )

        memory_output = torch.cat(outputs, dim=2)[:, :, pending_length:].transpose(1, 2).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.memory_width
        )
        memory_output = self._output_projection(memory_output, hidden_states)

        if past_key_values is not None and use_cache and update_cache:
            past_key_values.set_memory_state(self.layer_idx, cur_state, detach=True)
        return memory_output, cur_state

class TitansAttention(nn.Module):
    def __init__(self, config: TitansConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = TitansRotaryEmbedding(config)
        self.apply_rotary_pos_emb = apply_rotary_pos_emb

        if config.use_attention_convolution:
            self.q_conv = TitansDepthwiseConv1d(self.num_heads * self.head_dim, config.attention_qkv_conv_kernel)
            self.k_conv = TitansDepthwiseConv1d(
                self.num_key_value_heads * self.head_dim, config.attention_qkv_conv_kernel
            )
            self.v_conv = TitansDepthwiseConv1d(
                self.num_key_value_heads * self.head_dim, config.attention_qkv_conv_kernel
            )
        else:
            self.q_conv = self.k_conv = self.v_conv = None

    def _shape_q(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], tensor.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], tensor.shape[1], self.num_key_value_heads, self.head_dim).transpose(1, 2)

    def _project_tokens(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[TitansCache],
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if self.q_conv is not None:
            query = self.q_conv(query, past_key_values, self.layer_idx, "attention", "q", use_cache)
            key = self.k_conv(key, past_key_values, self.layer_idx, "attention", "k", use_cache)
            value = self.v_conv(value, past_key_values, self.layer_idx, "attention", "v", use_cache)
        return F.silu(query), F.silu(key), F.silu(value)

    def _project_prefix(self, prefix_states: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if prefix_states is None or prefix_states.shape[1] == 0:
            return None, None
        key = F.silu(self.k_proj(prefix_states))
        value = F.silu(self.v_proj(prefix_states))
        return self._shape_kv(key), self._shape_kv(value)

    def _attention_mask(
        self,
        batch_size: int,
        query_length: int,
        key_length: int,
        prefix_length: int,
        past_token_length: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        sliding_window: Optional[int],
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_positions = torch.arange(past_token_length, past_token_length + query_length, device=device)
        key_prefix = torch.full((prefix_length,), -1, device=device, dtype=torch.long)
        key_tokens = torch.arange(max(past_token_length + query_length, key_length - prefix_length), device=device)
        if key_tokens.shape[0] > key_length - prefix_length:
            key_tokens = key_tokens[-(key_length - prefix_length) :]
        key_positions = torch.cat([key_prefix, key_tokens], dim=0)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        prefix_allowed = key_positions.unsqueeze(0) < 0
        allowed = causal | prefix_allowed
        if sliding_window is not None and key_tokens.numel() > 0:
            window_allowed = key_positions.unsqueeze(0) >= (query_positions.unsqueeze(1) - sliding_window + 1)
            allowed = prefix_allowed | (allowed & window_allowed)
        mask = torch.zeros(query_length, key_length, device=device, dtype=torch.float32)
        mask = mask.masked_fill(~allowed, torch.finfo(torch.float32).min)
        mask = mask.view(1, 1, query_length, key_length).expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            token_key_length = key_length - prefix_length
            if attention_mask.shape[-1] == key_length:
                key_mask = attention_mask
            elif attention_mask.shape[-1] == token_key_length:
                prefix_mask = torch.ones(batch_size, prefix_length, device=device, dtype=attention_mask.dtype)
                key_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            elif attention_mask.shape[-1] == query_length and token_key_length == query_length:
                prefix_mask = torch.ones(batch_size, prefix_length, device=device, dtype=attention_mask.dtype)
                key_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            else:
                token_mask = attention_mask[:, -token_key_length:]
                if token_mask.shape[-1] < token_key_length:
                    pad = torch.ones(
                        batch_size,
                        token_key_length - token_mask.shape[-1],
                        device=device,
                        dtype=attention_mask.dtype,
                    )
                    token_mask = torch.cat([pad, token_mask], dim=-1)
                prefix_mask = torch.ones(batch_size, prefix_length, device=device, dtype=attention_mask.dtype)
                key_mask = torch.cat([prefix_mask, token_mask], dim=-1)
            mask = mask.masked_fill(key_mask[:, None, None, :].to(torch.bool).logical_not(), torch.finfo(torch.float32).min)
        if prefix_attention_mask is not None:
            blocked = F.pad(~prefix_attention_mask.to(torch.bool), (0, key_length - prefix_length), value=False)
            mask = mask.masked_fill(blocked[:, None], torch.finfo(torch.float32).min)
        return mask.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TitansCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        prefix_states: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        cache_attention: bool = True,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_length, _ = hidden_states.shape
        if position_ids is None:
            start = past_key_values.seqlen_offset if past_key_values is not None else 0
            position_ids = torch.arange(start, start + query_length, device=hidden_states.device).unsqueeze(0)

        query, key, value = self._project_tokens(hidden_states, past_key_values, use_cache and cache_attention)
        query = self._shape_q(query)
        key = self._shape_kv(key)
        value = self._shape_kv(value)

        query = l2norm(query, dim=-1, eps=self.config.l2_norm_eps)
        key = l2norm(key, dim=-1, eps=self.config.l2_norm_eps)
        prefix_key, prefix_value = self._project_prefix(prefix_states)
        prefix_length = 0 if prefix_key is None else prefix_key.shape[2]
        if prefix_length:
            # MAG's stream is P || x, so real token positions follow P.
            if self.config.variant == "mag":
                position_ids = position_ids + prefix_length
            prefix_positions = torch.arange(prefix_length, device=hidden_states.device)[None].expand(batch_size, -1)
            rope_positions = torch.cat([prefix_positions, position_ids.expand(batch_size, -1)], dim=1)
        else:
            rope_positions = position_ids
        # One call keeps dynamic RoPE's frequency basis consistent for prefix and tokens.
        cos, sin = self.rotary_emb(value, rope_positions)
        query, key = self.apply_rotary_pos_emb(query, key, cos[:, prefix_length:], sin[:, prefix_length:])
        if prefix_length:
            prefix_key = l2norm(prefix_key, dim=-1, eps=self.config.l2_norm_eps)
            prefix_cos, prefix_sin = cos[:, None, :prefix_length], sin[:, None, :prefix_length]
            rotary_dim = prefix_cos.shape[-1]
            prefix_rot = prefix_key[..., :rotary_dim]
            rotated = prefix_rot * prefix_cos + rotate_half(prefix_rot) * prefix_sin
            prefix_key = torch.cat([rotated, prefix_key[..., rotary_dim:]], dim=-1) if rotary_dim != self.head_dim else rotated

        past_token_length = 0
        if past_key_values is not None and use_cache and cache_attention:
            cached_key = past_key_values.key_cache[self.layer_idx]
            past_token_length = 0 if cached_key is None else cached_key.shape[2]
            key, value = past_key_values.update_attention(key, value, self.layer_idx, sliding_window)

        if prefix_key is not None:
            key = torch.cat([prefix_key, key], dim=2)
            value = torch.cat([prefix_value, value], dim=2)

        implementation = self.config._attn_implementation or "eager"
        uses_flash = is_flash_attention_requested(requested_attention_implementation=implementation)
        if output_attentions and implementation != "eager":
            logger.warning_once("Titans uses eager attention when output_attentions=True.")
            implementation = "eager"
            uses_flash = False
        elif uses_flash and (prefix_attention_mask is not None or (prefix_length and sliding_window is not None)):
            logger.warning_once(
                "Titans prefix attention cannot be represented by a FlashAttention sliding window. "
                "Using SDPA for this attention call to preserve the persistent/retrieved prefix mask."
            )
            implementation = "sdpa"
            uses_flash = False

        if uses_flash:
            if prefix_length and attention_mask is not None:
                attention_mask = F.pad(attention_mask, (prefix_length, 0), value=1)
            mask = flash_attention_mask(
                batch_size=batch_size,
                q_length=query_length,
                kv_length=key.shape[2],
                attention_mask=attention_mask,
            )
        else:
            mask = self._attention_mask(
                batch_size, query_length, key.shape[2], prefix_length, past_token_length,
                attention_mask, hidden_states.device, query.dtype, sliding_window, prefix_attention_mask,
            )
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query, key, value, mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=sliding_window,
            is_causal=True,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, query_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None


class TitansFusionGate(nn.Module):
    """Combine normalized attention and memory outputs, without the block residual."""

    def __init__(self, config: TitansConfig):
        super().__init__()
        self.memory_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.core_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate_bias = config.gate_bias

    def forward(self, core: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        core_n = self.core_norm(core)
        memory_n = self.memory_norm(memory)
        gate = torch.sigmoid(self.gate_proj(torch.cat([core_n, memory_n], dim=-1)) + self.gate_bias)
        return self.out_proj(gate * core_n + (1.0 - gate) * memory_n)


class TitansDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TitansConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.variant = config.variant
        self.memory = TitansNeuralMemory(config, layer_idx)
        self.self_attn = None if self.variant == "lmm" else TitansAttention(config, layer_idx)
        self.fusion = None if self.variant in ["lmm", "mal"] else TitansFusionGate(config)
        self.memory_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_norm = None if self.variant == "lmm" else TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TitansMLP(config)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        if config.persistent_memory_tokens > 0:
            self.persistent_memory = nn.Parameter(torch.empty(config.persistent_memory_tokens, config.hidden_size))
        else:
            self.persistent_memory = None

    def _persistent(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        if self.persistent_memory is None:
            return None
        return self.persistent_memory.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _with_persistent(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[TitansCache],
        use_cache: bool,
    ) -> tuple[torch.Tensor, int]:
        if use_cache and past_key_values is not None and past_key_values.seqlen_offset > 0:
            return hidden_states, 0
        persistent = self._persistent(hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
        if persistent is None:
            return hidden_states, 0
        return torch.cat([persistent, hidden_states], dim=1), persistent.shape[1]

    def _forward_lmm(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[TitansCache],
        use_cache: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        memory_input = self.memory_norm(hidden_states)
        memory_input, prefix_len = self._with_persistent(memory_input, past_key_values, use_cache)
        memory_output, _ = self.memory(memory_input, past_key_values=past_key_values, use_cache=use_cache)
        memory_output = memory_output[:, prefix_len:]
        hidden_states = residual + self.resid_dropout(memory_output)
        return hidden_states, None

    def _forward_mal(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[TitansCache],
        use_cache: bool,
        output_attentions: bool,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        memory_input = self.memory_norm(hidden_states)
        memory_input, prefix_len = self._with_persistent(memory_input, past_key_values, use_cache)
        memory_output, _ = self.memory(memory_input, past_key_values=past_key_values, use_cache=use_cache)
        if prefix_len:
            residual = torch.cat([memory_input[:, :prefix_len], residual], dim=1)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (self.config.persistent_memory_tokens, 0), value=1)
        hidden_states = residual + self.resid_dropout(memory_output)

        # MAL attends over M(P || x), including the transformed persistent tokens.
        if position_ids is not None:
            position_ids = position_ids + self.config.persistent_memory_tokens
            if prefix_len:
                prefix_positions = torch.arange(prefix_len, device=hidden_states.device)
                position_ids = torch.cat([prefix_positions[None].expand(hidden_states.shape[0], -1), position_ids], dim=1)
        residual = hidden_states
        attn_output, attn_weights = self.self_attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            sliding_window=self.config.sliding_window,
            **kwargs,
        )
        hidden_states = (residual + self.resid_dropout(attn_output))[:, prefix_len:]
        if attn_weights is not None:
            attn_weights = attn_weights[:, :, prefix_len:]
        return hidden_states, attn_weights

    def _forward_mag(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[TitansCache],
        use_cache: bool,
        output_attentions: bool,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        attn_input = self.attn_norm(hidden_states)
        persistent = self._persistent(hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
        if persistent is not None:
            persistent = self.attn_norm(persistent)
        attn_output, attn_weights = self.self_attn(
            attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            prefix_states=persistent,
            sliding_window=self.config.sliding_window,
            **kwargs,
        )
        memory_input, prefix_len = self._with_persistent(hidden_states, past_key_values, use_cache)
        memory_input = self.memory_norm(memory_input)
        memory_output, _ = self.memory(memory_input, past_key_values=past_key_values, use_cache=use_cache)
        memory_output = memory_output[:, prefix_len:]
        hidden_states = residual + self.resid_dropout(self.fusion(attn_output, memory_output))
        return hidden_states, attn_weights

    def _forward_mac_segment(
        self,
        segment: torch.Tensor,
        retrieval_input: torch.Tensor,
        state: dict[str, list[torch.Tensor]],
        persistent: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        output_attentions: bool,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]], Optional[torch.Tensor]]:
        # MAC: retrieve from the frozen past, attend to P || h || x, then
        # learn from attention's output and read the causally updated memory.
        historical = self.memory.retrieve(retrieval_input, state)
        prefix = historical if persistent is None else torch.cat([persistent, historical], dim=1)
        persistent_len = 0 if persistent is None else persistent.shape[1]
        size = segment.shape[1]
        historical_mask = torch.ones(size, size, device=segment.device, dtype=torch.bool).tril()
        historical_mask = historical_mask[None] & attention_mask[:, None].to(torch.bool)
        prefix_mask = F.pad(historical_mask, (persistent_len, 0), value=True)
        # Retrievals depend on their corresponding input tokens. An earlier
        # query must not see retrievals made from later tokens in this segment.
        attn_output, attn_weights = self.self_attn(
            self.attn_norm(segment),
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            prefix_states=self.attn_norm(prefix),
            prefix_attention_mask=prefix_mask,
            sliding_window=None,
            cache_attention=False,
            **kwargs,
        )
        memory_output, next_state = self.memory(
            self.memory_norm(attn_output),
            state=state,
            update=True,
            output_after_update=True,
            update_cache=False,
        )
        return self.fusion(attn_output, memory_output), next_state, attn_weights

    def _forward_mac(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[TitansCache],
        use_cache: bool,
        output_attentions: bool,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        segments = hidden_states
        local_mask = (
            attention_mask[:, -hidden_states.shape[1]:]
            if attention_mask is not None else torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long)
        )
        if position_ids is None:
            offset = past_key_values.seqlen_offset if past_key_values is not None else 0
            position_ids = torch.arange(offset, offset + hidden_states.shape[1], device=hidden_states.device)[None]
            position_ids = position_ids.expand(hidden_states.shape[0], -1)

        cached_segment = past_key_values is not None and use_cache
        pending_length = 0
        if cached_segment:
            pending = past_key_values.mac_segment_inputs[self.layer_idx]
            if pending is not None:
                pending_length = pending[0].shape[1]
                segments = torch.cat([pending[0], segments], dim=1)
                local_mask = torch.cat([pending[1], local_mask], dim=1)
                position_ids = torch.cat([pending[2], position_ids], dim=1)

        # Keep replay inputs raw. Retrieval and attention have separate
        # pre-norms, and replay must not normalize an already-normalized token.
        retrieval_inputs = self.memory_norm(segments)
        state = (
            past_key_values.mac_segment_states[self.layer_idx]
            if pending_length else self.memory._state_from_cache_or_init(retrieval_inputs, past_key_values)
        )
        persistent = self._persistent(hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
        outputs = []
        attn_collection = [] if output_attentions else None
        segment_size = self.config.mac_segment_size
        for start in range(0, segments.shape[1], segment_size):
            end = min(start + segment_size, segments.shape[1])
            segment_start_state = state
            output, state, attn_weights = self._forward_mac_segment(
                segment=segments[:, start:end],
                retrieval_input=retrieval_inputs[:, start:end],
                state=state,
                persistent=persistent,
                attention_mask=local_mask[:, start:end],
                position_ids=position_ids[:, start:end],
                output_attentions=output_attentions,
                **kwargs,
            )
            outputs.append(output)
            if output_attentions:
                attn_collection.append(attn_weights[:, :, max(0, pending_length - start):])

        if cached_segment:
            past_key_values.set_memory_state(self.layer_idx, state, detach=True)
            remainder = segments.shape[1] % segment_size
            past_key_values.mac_segment_inputs[self.layer_idx] = (
                tuple(value[:, -remainder:].detach().clone() for value in (segments, local_mask, position_ids))
                if remainder else None
            )
            past_key_values.mac_segment_states[self.layer_idx] = (
                {key: [value.detach() for value in values] for key, values in segment_start_state.items()}
                if remainder else None
            )
        hidden_states = residual + self.resid_dropout(torch.cat(outputs, dim=1)[:, pending_length:])
        if output_attentions:
            return hidden_states, tuple(attn_collection)
        return hidden_states, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TitansCache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.variant == "lmm":
            hidden_states, attn_weights = self._forward_lmm(hidden_states, past_key_values, use_cache)
        elif self.variant == "mal":
            hidden_states, attn_weights = self._forward_mal(
                hidden_states, attention_mask, position_ids, past_key_values, use_cache, output_attentions, **kwargs
            )
        elif self.variant == "mag":
            hidden_states, attn_weights = self._forward_mag(
                hidden_states, attention_mask, position_ids, past_key_values, use_cache, output_attentions, **kwargs
            )
        elif self.variant == "mac":
            hidden_states, attn_weights = self._forward_mac(
                hidden_states, attention_mask, position_ids, past_key_values, use_cache, output_attentions, **kwargs
            )
        else:  # pragma: no cover - config validation prevents this.
            raise ValueError(f"Unsupported Titans variant {self.variant!r}.")

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_dropout(hidden_states)
        return hidden_states, attn_weights


class TitansPreTrainedModel(PreTrainedModel):
    config_class = TitansConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TitansDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _supports_context_parallel = False

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        super()._init_weights(module)
        if isinstance(module, TitansNeuralMemory):
            for weight in module.weights:
                init.normal_(weight, mean=0.0, std=std)
            for bias in module.biases:
                init.zeros_(bias)
            if module.control_proj.bias is not None and not getattr(
                module.control_proj.bias, "_is_hf_initialized", False
            ):
                with torch.no_grad():
                    control_bias = module.control_proj.bias.view(module.num_heads, 3)
                    control_bias[:, 0].fill_(self.config.memory_alpha_bias)
                    control_bias[:, 1].fill_(self.config.memory_eta_bias)
                    control_bias[:, 2].fill_(self.config.memory_theta_bias)
        elif isinstance(module, TitansDecoderLayer) and module.persistent_memory is not None:
            init.normal_(module.persistent_memory, mean=0.0, std=std)


@dataclass
class TitansModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[TitansCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[Any, ...]] = None


class TitansModel(TitansPreTrainedModel):
    def __init__(self, config: TitansConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TitansDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _forward_padded(
        self,
        inputs_embeds: torch.Tensor,
        valid: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[TitansCache],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TitansModelOutputWithPast:
        """Run each unpadded stream so padding never advances memory or convolutions."""
        batch_size, seq_len, _ = inputs_embeds.shape
        if past_key_values is not None and past_key_values.batch_caches is not None:
            cache = past_key_values
            row_caches = cache.batch_caches
        else:
            row_caches = [
                past_key_values._select_batch(torch.tensor([row], device=inputs_embeds.device))
                if past_key_values is not None else None
                for row in range(batch_size)
            ]
            cache = TitansCache(self.config, batch_size, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            cache.seqlen_offset = past_key_values.seqlen_offset if past_key_values is not None else 0
        hidden_rows = []
        hidden_collections = [[] for _ in range(len(self.layers) + 1)] if output_hidden_states else None
        attention_collections = [[] for _ in self.layers] if output_attentions else None
        if position_ids is not None:
            position_ids = position_ids.expand(batch_size, -1)
        for row in range(batch_size):
            indices = valid[row].nonzero(as_tuple=True)[0]
            empty = inputs_embeds[row] * 0.0
            if indices.numel() == 0:
                hidden_rows.append(empty)
                if output_hidden_states:
                    for collection in hidden_collections:
                        collection.append(empty)
                if output_attentions:
                    for collection in attention_collections:
                        collection.append(None)
                continue
            result = self(
                inputs_embeds=inputs_embeds[row:row + 1].index_select(1, indices),
                position_ids=position_ids[row:row + 1].index_select(1, indices) if position_ids is not None else None,
                past_key_values=row_caches[row],
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs,
            )
            hidden_rows.append(empty.index_copy(0, indices, result.last_hidden_state[0]))
            row_caches[row] = result.past_key_values
            if output_hidden_states:
                for collection, value in zip(hidden_collections, result.hidden_states):
                    collection.append(empty.index_copy(0, indices, value[0]))
            if output_attentions:
                for collection, value in zip(attention_collections, result.attentions):
                    collection.append(value)
        if use_cache:
            cache.batch_caches = row_caches
            cache.seqlen_offset += seq_len
        return TitansModelOutputWithPast(
            last_hidden_state=torch.stack(hidden_rows),
            past_key_values=cache if use_cache else None,
            hidden_states=tuple(torch.stack(rows) for rows in hidden_collections) if output_hidden_states else None,
            attentions=tuple(tuple(rows) for rows in attention_collections) if output_attentions else None,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TitansCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_params: Optional[TitansCache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, TitansModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if cache_params is not None and past_key_values is None:
            past_key_values = cache_params

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if any(kwargs.get(name) is not None for name in ("cu_seq_lens_q", "cu_seq_lens_k")):
            raise ValueError("Titans recurrent memory does not support packed attention metadata. Use separate batch streams.")

        if self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape
        valid = attention_mask[:, -seq_len:].to(torch.bool) if attention_mask is not None else None
        if (valid is not None and not bool(valid.all())) or (
            past_key_values is not None and past_key_values.batch_caches is not None
        ):
            if valid is None:
                valid = torch.ones((batch_size, seq_len), dtype=torch.bool, device=inputs_embeds.device)
            outputs = self._forward_padded(
                inputs_embeds, valid, position_ids, past_key_values, use_cache, output_attentions, output_hidden_states, **kwargs
            )
            return outputs if return_dict else outputs.to_tuple()

        if use_cache and past_key_values is None:
            past_key_values = TitansCache(
                self.config,
                batch_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                model=self,
            )

        past_seen_tokens = past_key_values.seqlen_offset if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                dtype=torch.long,
                device=inputs_embeds.device,
            ).unsqueeze(0).expand(batch_size, -1)
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, past_seen_tokens + seq_len, dtype=torch.long, device=inputs_embeds.device
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weights = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            if output_attentions:
                all_attentions += (attn_weights,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if use_cache and past_key_values is not None:
            past_key_values.seqlen_offset += seq_len

        if not return_dict:
            return tuple(v for v in (hidden_states, past_key_values if use_cache else None, all_hidden_states, all_attentions) if v is not None)
        return TitansModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class TitansForCausalLM(TitansPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(self, config: TitansConfig):
        super().__init__(config)
        self.model = TitansModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TitansCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_params: Optional[TitansCache] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_params=cache_params,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(slice_logits_to_keep(hidden_states, logits_to_keep))

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "TitansCache",
    "TitansConfig",
    "TitansForCausalLM",
    "TitansModel",
    "TitansModelOutputWithPast",
    "TitansPreTrainedModel",
    "l2norm",
]
