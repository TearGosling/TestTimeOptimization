from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

# All necessary for now, manually doing a `modeling_X.py` file from transformers but
# later I can just use the transformers source tool.
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, logging
from transformers.utils.generic import maybe_autocast
from transformers.utils.import_utils import is_causal_conv1d_available

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

# TODO(TG): Implement fallback naive version of parallel scan later.
try:
    from accelerated_scan.scalar import scan
except ImportError:
    raise ImportError(
        "Please install the accelerated-scan package: https://github.com/proger/accelerated-scan"
        "You can install it via `pip install accelerated_scan`."
    )

try:
    from .configuration_titans import TitansConfig
except ImportError:
    from configuration_titans import TitansConfig

logger = logging.get_logger(__name__)

class TitansRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        TitansRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class TitansRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: TitansConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: TitansConfig | None = None,
        device: Optional[torch.device] = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def silu_backward(x):
    """SiLU backward function."""
    sig = torch.sigmoid(x)
    return sig + (F.silu(x) * (1.0 - sig))

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def parallel_scan(
    chunk_decays: torch.Tensor,
    chunk_values: torch.Tensor,
    init_value: torch.Tensor,
) -> torch.Tensor:
    """
    Performs an associative scan over the input values with the given decays.

    Uses 4D tensors for memory efficiency (activation-space, not weight-space).

    Args:
        chunk_decays: [batch, num_heads, seq_len, 1] decay factors per timestep
        chunk_values: [batch, num_heads, seq_len, dim_size] values to accumulate
        init_value: [batch, num_heads, 1, dim_size] initial carry value

    Returns:
        Scanned values [batch, num_heads, seq_len, dim_size]
    """
    batch_size, num_heads, seq_len, dim_size = chunk_values.shape

    # Pad decays with 1.0 for the initial position (identity for multiplication)
    decays = F.pad(chunk_decays, (0, 0, 1, 0), value=1.0)

    # Prepend init_value to values
    values = torch.cat([init_value, chunk_values], dim=2)

    # Expand decay to match value shape
    decays = decays.expand(-1, -1, -1, dim_size)

    # Reshape for accelerated-scan which expects 3D tensors [B, C, L]
    decays = decays.transpose(-2, -1).reshape(batch_size, num_heads * dim_size, seq_len + 1)
    values = values.transpose(-2, -1).reshape(batch_size, num_heads * dim_size, seq_len + 1)

    # Run scan and chop off the initial padding
    scanned = scan(decays, values)[:, :, 1:]

    return scanned.view(batch_size, num_heads, dim_size, seq_len).transpose(-2, -1).contiguous()

"""
TODO(TG): Implement TitansCache.
NOTES:
- Cache must store states, grads and momentum for every layer's memory module, as well as the states of the convolutions in those layers.
- The cache structure for the attention component will differ depending on the selected variant.
    - MAC (variant="mac"): Will need to cache the long-term memory tokens (Q vector of last segment) and the KV vectors for the *last segment only* of full causal attention.
    It will require noting the number of chunked attention segments as well.
    - MAG (variant="mag") and MAL (variant="mal"): Will need a KV cache for sliding-window attention rather than standard attention.
    - LMM (variant="lmm"): No attention cache at all.
- Dicts/attributes:
    - conv_states: Dict[str, List[torch.Tensor]]: Stores the convolution states for each layer's convolution modules.
"""

class TitansCache:
    pass

# Convolution.
class TitansConv(nn.Module):
    def __init__(self, config: TitansConfig, layer_idx: int, conv_name: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_mem_heads

        # K/V convs may be smaller than hidden_size if num_kv_heads < num_attention_heads
        if "attn" in conv_name:
            if conv_name == "attn_q":
                channel_size = config.hidden_size
            else:
                channel_size = self.head_dim * config.num_key_value_heads
        else:
            channel_size = config.hidden_size

        self.conv_name = conv_name
        self.conv = nn.Conv1d(
            in_channels=channel_size,
            out_channels=channel_size,
            bias=True,
            kernel_size=config.conv_kernel,
            groups=channel_size,
            padding=config.conv_kernel - 1,
        )

    def forward(self, hidden_states: torch.Tensor, cache: TitansCache | None = None) -> torch.Tensor:
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, hidden_size, seq_len]
        bsz, seq_len, n_heads, head_dim = hidden_states.size()

        hidden_states = hidden_states.reshape(bsz, seq_len, n_heads * head_dim).transpose(1, 2)
        if cache is not None:
            current_conv_state = cache.conv_states[self.conv_name][self.layer_idx]
        
        # Copied from TTT - this probably can be optimized/cleaned up further.
        if causal_conv1d_fn is None:
            if cache is not None:
                if cache.seqlen_offset > 0:
                    new_conv_state = current_conv_state
                    new_conv_state = torch.roll(new_conv_state, shifts=-1, dims=-1)
                    new_conv_state[:, :, -1] = hidden_states[:, :, 0]
                    current_conv_state.copy_(new_conv_state)

                    hidden_states = torch.sum(new_conv_state * self.conv.weight[:, 0, :], dim=-1)
                    hidden_states += self.conv.bias
                    hidden_states = hidden_states.unsqueeze(-1)
                else:
                    new_conv_state = F.pad(
                        hidden_states,
                        (self.config.conv_kernel - hidden_states.shape[-1], 0),
                    )
                    current_conv_state.copy_(new_conv_state)
            hidden_states = self.conv(hidden_states)[..., :seq_len]
            hidden_states = F.silu(hidden_states)
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            if cache is not None and cache.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    current_conv_state,
                    conv_weights,
                    self.conv.bias,
                    None,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache is not None:
                    conv_states = F.pad(
                        hidden_states,
                        (self.config.conv_kernel - hidden_states.shape[-1], 0),
                    )
                    current_conv_state.copy_(conv_states)
                hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv.bias, activation="silu")

        # [batch_size, hidden_size, seq_len] -> [batch_size, num_heads, seq_len, head_dim]
        # TODO(TG): Optimize reshape/transposes.
        hidden_states = hidden_states.reshape(bsz, n_heads, head_dim, seq_len).transpose(-1, -2)
        return hidden_states

# TODO(TG): Implement TitansAttention later for non-LMM variants.
class TitansAttention:
    pass

class TitansMemoryModule(nn.Module):
    def __init__(self, config: TitansConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_mem_heads
        self.chunk_size = config.chunk_size

        # TODO(TG): No GQA support for memory module for now.
        self.q_proj = nn.Linear(config.hidden_size, config.num_mem_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_mem_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_mem_heads * self.head_dim, bias=False)

        # NOTE(TG): Titans paper states they use L2 normalization for QK, but lucidrains and convention
        # uses RMSnorm. I'm providing the option for either.
        self.q_norm = TitansRMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps) if config.rms_qk_norm else lambda x: F.normalize(x, p=2.0, dim=-1)
        self.k_norm = TitansRMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps) if config.rms_qk_norm else lambda x: F.normalize(x, p=2.0, dim=-1)

        self.conv_q = TitansConv(config, layer_idx, conv_name="mem_q")
        self.conv_k = TitansConv(config, layer_idx, conv_name="mem_k")
        self.conv_v = TitansConv(config, layer_idx, conv_name="mem_v")

        # Gate projections to memory decay (alpha), surprise decay (eta) and learning rate (theta)
        self.mem_decay_proj = nn.Linear(config.hidden_size, config.num_mem_heads, bias=True)
        self.surprise_decay_proj = nn.Linear(config.hidden_size, config.num_mem_heads, bias=True)
        self.lr_proj = nn.Linear(config.hidden_size, config.num_mem_heads, bias=True)

        mem_intermediate_size = int(self.head_dim * config.mem_expansion_factor)
        # MLP acting as our memory module.
        self.W1 = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=config.initializer_range,
                size=(config.num_mem_heads, self.head_dim, mem_intermediate_size),
            )
        )
        self.W2 = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=config.initializer_range,
                size=(config.num_mem_heads, mem_intermediate_size, self.head_dim),
            )
        )

        # Contains current state of the memory module parameters and their momentums.
        # Re-initialized at every forward pass if cache is None.
        self.params_dict = None

    def reset_params(self, batch_size: int) -> None:
        """
        Resets the memory module parameters to initial states.

        Uses activation-space momentum (4D tensors) instead of weight-space (5D)
        for O(L*D) memory scaling instead of O(L*D1*D2).
        """
        mem_intermediate_size = int(self.head_dim * self.config.mem_expansion_factor)
        self.params_dict = {
            # Weights: [B, H, head_dim, intermediate] and [B, H, intermediate, head_dim]
            "W1": self.W1.unsqueeze(0).expand(batch_size, -1, -1, -1).clone(),
            "W2": self.W2.unsqueeze(0).expand(batch_size, -1, -1, -1).clone(),
            # Activation-space momentum: [B, H, 1, D] - NOT weight-space [B, H, D1, D2]
            "W1_surprise": torch.zeros(batch_size, self.config.num_mem_heads, 1, mem_intermediate_size,
                                      device=self.W1.device, dtype=self.W1.dtype),
            "W2_surprise": torch.zeros(batch_size, self.config.num_mem_heads, 1, self.head_dim,
                                      device=self.W2.device, dtype=self.W2.dtype),
            # Input-side momentum for outer product weight updates
            "key_momentum": torch.zeros(batch_size, self.config.num_mem_heads, 1, self.head_dim,
                                       device=self.W1.device, dtype=self.W1.dtype),
            "X1_momentum": torch.zeros(batch_size, self.config.num_mem_heads, 1, mem_intermediate_size,
                                      device=self.W1.device, dtype=self.W1.dtype),
        }

    def retrieve(self, query_states: torch.Tensor) -> torch.Tensor:
        """
        Retrieves information from the memory module.
        """
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        W1 = self.params_dict["W1"]
        W2 = self.params_dict["W2"]

        # Pass through memory MLP
        mem_states = query_states @ W1  # [batch, num_heads, seq_len, mem_intermediate_size]
        mem_states = F.silu(mem_states)
        mem_states = mem_states @ W2  # [batch, num_heads, seq_len, head_dim]

        return mem_states.transpose(-2, -1).reshape(batch_size, seq_len, num_heads * head_dim)
    
    def update(self, hidden_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, cache: TitansCache | None = None) -> None:
        """
        Updates the memory module with new information.

        Key optimization: accumulate activation-space gradients via scan (4D tensors),
        then apply outer products only at chunk boundaries to update weights.
        This gives O(L*D) memory instead of O(L*D1*D2).
        """
        W1 = self.params_dict["W1"]
        W2 = self.params_dict["W2"]
        W1_surprise = self.params_dict["W1_surprise"]
        W2_surprise = self.params_dict["W2_surprise"]
        key_momentum = self.params_dict["key_momentum"]
        X1_momentum = self.params_dict["X1_momentum"]

        # Get decays and LR from hidden states [B, L, H] -> [B, H, L, 1]
        mem_decay = 1.0 - torch.sigmoid(self.mem_decay_proj(hidden_states))
        mem_decay = mem_decay.transpose(-2, -1).unsqueeze(-1)
        surprise_decay = torch.sigmoid(self.surprise_decay_proj(hidden_states))
        surprise_decay = surprise_decay.transpose(-2, -1).unsqueeze(-1)
        lr = torch.sigmoid(self.lr_proj(hidden_states))
        lr = lr.transpose(-2, -1).unsqueeze(-1)

        # Forward pass through memory MLP with keys
        Z1 = key_states @ W1  # [B, H, L, intermediate]
        X1 = F.silu(Z1)
        mems = X1 @ W2  # [B, H, L, head_dim]

        # Gradient of L2 loss w.r.t. activations (activation-space, not weight-space)
        grad_wrt_mems = (mems - value_states)  # [B, H, L, head_dim]
        grad_wrt_Z1 = (grad_wrt_mems @ W2.transpose(-2, -1)) * silu_backward(Z1)  # [B, H, L, intermediate]

        # Scale by learning rate (negative for gradient descent)
        grad_wrt_mems = -lr * grad_wrt_mems
        grad_wrt_Z1 = -lr * grad_wrt_Z1

        # Accumulate activation-space gradients via parallel scan (surprise momentum)
        # This is O(L*D) instead of O(L*D1*D2)
        W1_surprise_new = parallel_scan(surprise_decay, grad_wrt_Z1, init_value=W1_surprise)
        W2_surprise_new = parallel_scan(surprise_decay, grad_wrt_mems, init_value=W2_surprise)

        # Also accumulate the "input" side for outer product
        key_momentum_new = parallel_scan(surprise_decay, key_states, init_value=key_momentum)
        X1_momentum_new = parallel_scan(surprise_decay, X1, init_value=X1_momentum)

        # Weight update: outer product of accumulated inputs and gradients at chunk end
        key_accum = key_momentum_new[:, :, -1:, :]   # [B, H, 1, head_dim]
        X1_accum = X1_momentum_new[:, :, -1:, :]     # [B, H, 1, intermediate]
        grad_Z1_accum = W1_surprise_new[:, :, -1:, :]   # [B, H, 1, intermediate]
        grad_mem_accum = W2_surprise_new[:, :, -1:, :]  # [B, H, 1, head_dim]

        # Compute weight deltas via outer products
        dW1 = key_accum.transpose(-2, -1) @ grad_Z1_accum   # [B, H, head_dim, intermediate]
        dW2 = X1_accum.transpose(-2, -1) @ grad_mem_accum   # [B, H, intermediate, head_dim]

        # Apply memory decay and update weights
        avg_mem_decay = mem_decay.mean(dim=2, keepdim=True).squeeze(2)  # [B, H, 1]
        W1 = avg_mem_decay.unsqueeze(-1) * W1 + dW1
        W2 = avg_mem_decay.unsqueeze(-1) * W2 + dW2

        # Update params dict
        self.params_dict["W1"] = W1
        self.params_dict["W2"] = W2
        self.params_dict["W1_surprise"] = W1_surprise_new[:, :, -1:, :]
        self.params_dict["W2_surprise"] = W2_surprise_new[:, :, -1:, :]
        self.params_dict["key_momentum"] = key_momentum_new[:, :, -1:, :]
        self.params_dict["X1_momentum"] = X1_momentum_new[:, :, -1:, :]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: TitansCache | None = None,
    ) -> torch.Tensor:
        """
        Performs a retrieval step and an update step on the memory module in one go.
        This forward function is only used for the MAL and LMM variants of Titans, with
        the others by necessity having to need self.retrieve and self.update called separately.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project all queries, keys, values.
        # Right now, I can't do it in a tiled manner because of the convs.
        # TODO(TG): Add tiled conv support later.
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.conv_q(self.q_norm(query_states), cache=cache)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.conv_k(self.k_norm(key_states), cache=cache)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.conv_v(value_states, cache=cache)

        # Use separate output buffer to avoid inplace modification of hidden_states
        # which would break gradient computation during backprop
        output_chunks = []

        # Split hidden states into chunks along the sequence dimension for update processing.
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min((chunk_idx + 1) * self.chunk_size, seq_len)

            chunk_hidden_states = hidden_states[:, chunk_start:chunk_end, :]
            chunk_query_states = query_states[:, :, chunk_start:chunk_end, :]
            chunk_key_states = key_states[:, :, chunk_start:chunk_end, :]
            chunk_value_states = value_states[:, :, chunk_start:chunk_end, :]

            retrieved_chunk = self.retrieve(chunk_query_states)
            self.update(chunk_hidden_states, chunk_key_states, chunk_value_states, cache=cache)
            output_chunks.append(retrieved_chunk)

        # Concatenate all chunks
        return torch.cat(output_chunks, dim=1)

class TitansMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class TitansDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TitansConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = None # Will change when implementing non-LMM variants.
        self.memory = TitansMemoryModule(config, layer_idx)
        self.mlp = TitansMLP(config)
        self.input_layernorm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_layernorm = None
        self.post_memory_layernorm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: TitansCache | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self-Attention block would go here for non-LMM variants.
        hidden_states = self.memory(hidden_states=hidden_states, cache=cache)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_memory_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
class TitansPreTrainedModel(PreTrainedModel):
    config: TitansConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TitansDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": TitansDecoderLayer,
    }

class TitansModel(TitansPreTrainedModel):
    def __init__(self, config: TitansConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [TitansDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def reset_params(self, batch_size: int) -> None:
        """
        Resets the memory module parameters for all layers.
        """
        for layer in self.layers:
            layer.memory.reset_params(batch_size)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: TitansCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            # TODO(TG): Implement cache later.
            use_cache = False

        if not use_cache and cache is None:
            # Reset memory module params if no cache is provided.
            self.reset_params(batch_size=inputs_embeds.size(0))

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        hidden_states = self.final_norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)
    
class TitansForCausalLM(TitansPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: TitansConfig):
        super().__init__(config)
        self.model = TitansModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: TitansCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
