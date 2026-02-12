import math

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._pytree import tree_map

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

from .configuration_titans import TitansConfig

logger = logging.get_logger(__name__)

# TODO(TG): Create utils file later for these common modules.
def ln_fwd(x, gamma, beta, eps=1e-6):
    """
    Batch forward for LayerNorm.
    """
    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y

def ln_fused_l2_bwd(x, residual, pred, gamma, beta, eps=1e-6):
    """
    Batch backward for LayerNorm fused with L2 loss.
    """
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    y = y + residual # Residual

    # Derivative of L2 loss wrt activations.
    # TTT is cheeky by just having their loss function be 1/2 * ||mem - V||^2, thus eliminating
    # 2, but I don't know if Titans authors did this so I'll keep the 2 here for correctness.
    grad_output = 2.0 * (y - pred)
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z

def associative_scan_5d(gate, token, initial_state=None):
    """
    Vibe-coded scan but I've verified it is equivalent to the Triton scan fn.
    Performs a parallel associative scan (prefix sum over a semiring) for the 
    recurrence: x[t] = gate[t] * x[t-1] + token[t]
    
    This implementation uses the Hillis-Steele (recursive doubling) algorithm,
    which is efficient on GPUs with O(log L) steps.

    Args:
        gate:  Tensor of shape [B, H, L, 1, 1]
        token: Tensor of shape [B, H, L, D1, D2]
        initial_state: Optional Tensor of shape [B, H, D1, D2]
                       representing x[-1].

    Returns:
        Tensor of shape [B, H, L, D1, D2] representing the sequence x[t].
    """
    # 1. Validation and Setup
    # We clone the inputs to avoid in-place modifications to the original tensors
    # and to set up our accumulation buffers.
    # Dimensions: 0:Batch, 1:Heads, 2:Seq_Len, 3:Dim1, 4:Dim2
    curr_g = gate  # Ensure gate has shape [B, H, L, 1, 1]
    curr_u = token

    B, H, L, D1, D2 = token.shape
    
    # Calculate number of doubling steps required: ceil(log2(L))
    num_steps = int(math.ceil(math.log2(L)))

    # 2. The Associative Scan (Hillis-Steele Algorithm)
    # At every step k, we combine element i with element (i - 2^k).
    # The binary operator is: (a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2)
    # where 'a' is the gate (multiplicative) and 'b' is the token (additive).
    
    for i in range(num_steps):
        distance = 1 << i  # 2^i
        
        # We perform the operation only on the part of the sequence that has 
        # a valid predecessor 'distance' steps back.
        
        # Slice for the "current" elements (indices distance to L)
        g_now = curr_g[:, :, distance:, :, :]
        u_now = curr_u[:, :, distance:, :, :]
        
        # Slice for the "previous" elements (indices 0 to L-distance)
        g_prev = curr_g[:, :, :-distance, :, :]
        u_prev = curr_u[:, :, :-distance, :, :]
        
        # New U: (g_now * u_prev + u_now)
        u_new_part = g_now * u_prev + u_now
        # New G: (g_now * g_prev)
        g_new_part = g_now * g_prev
        
        # Concatenate the untouched prefix with the updated suffix
        # This creates a NEW tensor, allowing the old one to be freed if not needed
        # (autograd handles the graph connections cleanly)
        curr_u = torch.cat([curr_u[:, :, :distance, :, :], u_new_part], dim=2)
        curr_g = torch.cat([curr_g[:, :, :distance, :, :], g_new_part], dim=2)

    # 3. Handle Initial State
    # After the loop, curr_u contains the scan assuming x[-1] = 0.
    # curr_g contains the cumulative product of gates (g_0...g_t).
    # The actual result is: scan_result + cumprod_gates * initial_state
    
    if initial_state is not None:
        # initial_state: [B, H, 1, D1, D2]
        # curr_g:        [B, H, L, 1, 1]
        # Broadcasting handles the dimension expansion automatically.
        final_result = curr_u + (curr_g * initial_state)
    else:
        final_result = curr_u
        
    return final_result

def sequential_scan(f, init, xs, checkpoint_group=0):
    """
    Fixed sequential scan that handles dicts correctly with checkpointing
    and avoids side-effects on captured variables.
    """
    # Helper to unpack dicts to tuples (for checkpointing)
    def to_tuple(x):
        keys = sorted(x.keys())
        return tuple(x[k] for k in keys), keys

    def from_tuple(tup, keys):
        return {k: v for k, v in zip(keys, tup)}

    # Prepare inputs
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    carry = init
    
    # List to collect outputs (we will stack them later)
    # We cannot write to 'out' tensor in-place efficiently if we want to be clean with autograd
    # unless we carefully manage it. Collecting into a list is safer for memory in this context.
    outputs = []

    def scan_fn(carry_tup, carry_keys, start, end):
        # Reconstruct carry dict
        curr_carry = from_tuple(carry_tup, carry_keys)
        chunk_outputs = []
        
        for i in range(start, end):
            # Slice input
            if isinstance(xs, dict):
                x_i = {k: v[i] for k, v in xs.items()}
            else:
                x_i = [x[i] for x in xs]
            
            curr_carry, y = f(curr_carry, x_i)
            chunk_outputs.append(y)
            
        # Stack chunk outputs: [chunk_len, ...]
        stacked_out = torch.stack(chunk_outputs, dim=0)
        
        # Return new carry tuple AND the outputs for this chunk
        new_carry_tup, _ = to_tuple(curr_carry)
        return new_carry_tup, stacked_out

    curr_idx = 0
    while curr_idx < num_items:
        # Determine chunk size for this step
        if checkpoint_group > 0:
            step_size = min(checkpoint_group, num_items - curr_idx)
        else:
            step_size = num_items # Run all at once
            
        end_idx = curr_idx + step_size
        
        carry_tup, carry_keys = to_tuple(carry)
        
        if checkpoint_group > 0:
            # Checkpoint requires inputs to be tensors. 
            # We pass carry tensors individually (via unpacking) to ensure grad flows.
            # scan_fn returns (new_carry_tuple, output_tensor)
            new_carry_tup, chunk_out = torch.utils.checkpoint.checkpoint(
                scan_fn, 
                carry_tup, 
                carry_keys, 
                curr_idx, 
                end_idx, 
                use_reentrant=False
            )
        else:
            new_carry_tup, chunk_out = scan_fn(carry_tup, carry_keys, curr_idx, end_idx)
            
        # Update carry
        carry = from_tuple(new_carry_tup, carry_keys)
        
        # Collect output
        outputs.append(chunk_out)
        curr_idx = end_idx

    # Concatenate all outputs along time dimension (dim 0 here, because we stacked locally)
    # output shape: [seq_len, batch, ...]
    final_output = torch.cat(outputs, dim=0)
    return carry, final_output

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
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_mem_heads

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

def old_parallel_scan(
    decays: torch.Tensor,
    values: torch.Tensor,
    init_value: torch.Tensor,
) -> torch.Tensor:
    """
    Performs an associative scan over the input values with the given decays.
    Note that recurrence is computed with an initial value belonging to the previous chunk,
    necessitating supplying that too.
    """
    batch_size, num_heads, seq_len, dim1_size, dim2_size = values.shape
    # The third-party scan implementation does not allow specifying an initial value (default 0),
    # so we do the first step of the recurrence manually here.
    init_value = init_value.unsqueeze(2)  # [batch_size, num_heads, 1, dim1_size, dim2_size]
    decays = decays.expand_as(values)
    modified_first_term = (decays[:, :, 0:1] * init_value) + values[:, :, 0:1]
    values_new = torch.cat([modified_first_term, values[:, :, 1:]], dim=2)

    # Merge batch_size with heads and dim sizes for scan.
    decays = decays.reshape(batch_size * num_heads, seq_len, dim1_size * dim2_size).transpose(-2, -1).contiguous()
    values_new = values_new.reshape(batch_size * num_heads, seq_len, dim1_size * dim2_size).transpose(-2, -1).contiguous()
    # Run scan
    scanned = scan(decays, values_new)
    return scanned.reshape(batch_size, num_heads, dim1_size, dim2_size, seq_len).permute(0, 1, 4, 2, 3)

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
        self.num_heads = config.num_mem_heads
        self.head_dim = config.hidden_size // config.num_mem_heads
        self.chunk_size = config.chunk_size

        # TODO(TG): No GQA support for memory module for now, nor specifying the head_dim manually.
        self.q_proj = nn.Linear(config.hidden_size, config.num_mem_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_mem_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_mem_heads * self.head_dim, bias=False)

        # NOTE(TG): Titans paper states they use L2 normalization for QK, but lucidrains and convention
        # uses RMSnorm. I'm providing the option for either.
        l2_norm = lambda x: F.normalize(x, p=2.0, dim=-1)
        self.q_norm = TitansRMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps) if config.rms_qk_norm else l2_norm
        self.k_norm = TitansRMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps) if config.rms_qk_norm else l2_norm

        self.conv_q = TitansConv(config, layer_idx, conv_name="mem_q")
        self.conv_k = TitansConv(config, layer_idx, conv_name="mem_k")
        self.conv_v = TitansConv(config, layer_idx, conv_name="mem_v")

        # Gate projections to memory decay (alpha), surprise decay (eta) and learning rate (theta)
        self.mem_decay_proj = nn.Linear(config.hidden_size, config.num_mem_heads, bias=False)
        self.surprise_decay_proj = nn.Linear(config.hidden_size, config.num_mem_heads, bias=False)
        self.lr_proj = nn.Linear(config.hidden_size, config.num_mem_heads, bias=False)

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
        # LayerNorm for memory output.
        self.norm_w = nn.Parameter(torch.ones(config.num_mem_heads, self.head_dim))
        self.norm_b = nn.Parameter(torch.zeros(config.num_mem_heads, self.head_dim))
    
    def _reshape_pytree(self, t: torch.Tensor) -> torch.Tensor:
        """
        Reshaping function to prepare tensors for scan.
        """
        # control signals shape: [batch_size, seq_len, hidden_size]
        # qkv shape: [batch_size, num_heads, seq_len, head_dim]
        is_full_chunk = t.size(-2) % self.chunk_size == 0
        if is_full_chunk:
            if t.dim() == 3:
                t = t.unfold(1, self.chunk_size, self.chunk_size).permute(1, 0, 3, 2)  # [num_chunks, batch_size, chunk_size, hidden_size]
            elif t.dim() == 4:
                t = t.unfold(2, self.chunk_size, self.chunk_size).permute(2, 0, 1, 4, 3)  # [num_chunks, batch_size, num_heads, head_dim, chunk_size]
            else:
                raise ValueError(f"Unexpected tensor dimension {t.dim()} during pytree reshape")
        else:
            t = t.unsqueeze(0)
        return t

    def chunk_forward(
        self,
        params_dict: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor]
    ) -> tuple[dict[torch.Tensor], torch.Tensor]:
        query_states = inputs["query_states"]
        key_states = inputs["key_states"]
        value_states = inputs["value_states"]

        # Shape shift: [batch_size, seq_len, num_heads] -> [batch_size, num_heads, seq_len]
        mem_decay = inputs["mem_decay"].transpose(-2, -1)
        surprise_decay = inputs["surprise_decay"].transpose(-2, -1)[..., None, None] # Extra 2 dims required for parallel scan
        lr = inputs["lr"].transpose(-2, -1)

        # LR scaling is required for stable training.
        # Titans does not say how LR is scaled.
        # Lucidrains just multiplies by 0.1 or 0.01, so we'll do the same for now.
        lr = 0.01 * lr

        W1 = params_dict["W1"]
        W2 = params_dict["W2"]
        W1_surprise = params_dict["W1_surprise"]
        W2_surprise = params_dict["W2_surprise"]
        # Norm which gets us to grads wrt activations.
        ln_weight = self.norm_w.reshape(1, query_states.size(1), 1, self.head_dim)
        ln_bias = self.norm_b.reshape(1, query_states.size(1), 1, self.head_dim)

        # 1. Query retrieval from memory.
        Q_W1 = query_states @ W1
        Q_Z1 = F.silu(Q_W1)
        Q_W2 = Q_Z1 @ W2
        # layernorm
        upd_query_states = ln_fwd(Q_W2, ln_weight, ln_bias)

        # Residual, assuming this based on TTT and other papers.
        upd_query_states = query_states + upd_query_states

        # 2. Update parameter dictionary
        
        # Beta calculation as per Titans paper.
        betas = torch.cumprod(mem_decay, dim=-1)
        beta_T = betas[:, :, -1:]
        mem_decay_factors = beta_T / (betas + 1e-8)
        
        effective_lr = (lr * mem_decay_factors).unsqueeze(-1)

        # Keys fed through memory module
        # Pass through memory MLP
        Z1 = key_states @ W1  # [batch, num_heads, seq_len, mem_intermediate_size]
        X1 = F.silu(Z1)
        mems = X1 @ W2  # [batch, num_heads, seq_len, head_dim]
        
        grad_wrt_mems = ln_fused_l2_bwd(mems, key_states, value_states, ln_weight, ln_bias)
        grad_wrt_Z1 = grad_wrt_mems @ W2.transpose(-2, -1) * silu_backward(Z1)  # [batch, num_heads, seq_len, mem_intermediate_size]

        # Compute gradients for W1 and W2.
        grad_W2_t = -1.0 * torch.einsum('bhsi,bhsd,bhsi->bhsid', X1, grad_wrt_mems, effective_lr)
        grad_W1_t = -1.0 * torch.einsum('bhsd,bhsi,bhsi->bhsdi', key_states, grad_wrt_Z1, effective_lr)

        # Calculate new surprises via. associative scans.
        new_W1_surprise = associative_scan_5d(surprise_decay, grad_W1_t, initial_state=W1_surprise.unsqueeze(2)) # [batch_size, num_heads, seq_len, head_dim, mem_intermediate_size]
        new_W2_surprise = associative_scan_5d(surprise_decay, grad_W2_t, initial_state=W2_surprise.unsqueeze(2)) # [batch_size, num_heads, seq_len, head_dim, mem_intermediate_size]

        # Update weights
        new_W1 = W1 + new_W1_surprise[:, :, -1]
        new_W2 = W2 + new_W2_surprise[:, :, -1]
        new_W1_surprise = new_W1_surprise[:, :, -1]
        new_W2_surprise = new_W2_surprise[:, :, -1]

        # Update params dict with new weights and surprises at end of the chunk.
        new_params_dict = {
            "W1": new_W1,
            "W2": new_W2,
            "W1_surprise": new_W1_surprise,
            "W2_surprise": new_W2_surprise,
        }
        return new_params_dict, upd_query_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: TitansCache | None = None,
    ) -> torch.Tensor:
        """
        Performs a retrieval step and an update step on the memory module in one go.
        This forward function is only used for the MAL and LMM variants of Titans, with
        the others by necessity having to need self.retrieve and self.update called separately.
        """
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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        # Control signals projection.
        mem_decay = 1.0 - torch.sigmoid(self.mem_decay_proj(hidden_states)) # (1 - alpha) in Titans paper.
        surprise_decay = torch.sigmoid(self.surprise_decay_proj(hidden_states))
        lr = torch.sigmoid(self.lr_proj(hidden_states))

        # Split hidden states into chunks along the sequence dimension for sequential states.
        output_hidden_states = []
        num_full_chunks = hidden_states.size(1) // self.chunk_size
        remainder_len = hidden_states.size(1) % self.chunk_size

        # Process full chunks with sequential scan.
        params_dict = {
            "W1": torch.tile(self.W1.unsqueeze(0), dims=(input_shape[0], 1, 1, 1)),
            "W2": torch.tile(self.W2.unsqueeze(0), dims=(input_shape[0], 1, 1, 1)),
        }
        params_dict["W1_surprise"] = torch.zeros_like(params_dict["W1"])
        params_dict["W2_surprise"] = torch.zeros_like(params_dict["W2"])

        if num_full_chunks > 0:
            # Allocate empty output tensor.
            full_inputs = {
                "query_states": query_states[:, :, :num_full_chunks * self.chunk_size, :],
                "key_states": key_states[:, :, :num_full_chunks * self.chunk_size, :],
                "value_states": value_states[:, :, :num_full_chunks * self.chunk_size, :],
                "mem_decay": mem_decay[:, :num_full_chunks * self.chunk_size, :],
                "surprise_decay": surprise_decay[:, :num_full_chunks * self.chunk_size, :],
                "lr": lr[:, :num_full_chunks * self.chunk_size, :],
            }
            full_inputs = tree_map(lambda x: self._reshape_pytree(x), full_inputs)
            params_dict, output_full = sequential_scan(
                self.chunk_forward,
                init=params_dict,
                xs=full_inputs,
                checkpoint_group=self.config.scan_checkpoint_group_size if self.training else 0,
            )
            # Reshape output back to [batch_size, seq_len, hidden_size]
            output_full = output_full.permute(1, 0, 2, 3, 4).reshape(input_shape[0], num_full_chunks * self.chunk_size, -1)
            output_hidden_states.append(output_full)
        if remainder_len > 0:
            remainder_inputs = {
                "query_states": query_states[:, :, -remainder_len:, :],
                "key_states": key_states[:, :, -remainder_len:, :],
                "value_states": value_states[:, :, -remainder_len:, :],
                "mem_decay": mem_decay[:, -remainder_len:, :],
                "surprise_decay": surprise_decay[:, -remainder_len:, :],
                "lr": lr[:, -remainder_len:, :],
            }
            remainder_inputs = tree_map(lambda x: self._reshape_pytree(x), remainder_inputs)
            _, output_remainder = sequential_scan(
                self.chunk_forward,
                init=params_dict,
                xs=remainder_inputs,
                checkpoint_group=self.config.scan_checkpoint_group_size if self.training else 0,
            )
            output_remainder = output_remainder.squeeze(0).permute(0, 2, 1, 3).reshape(input_shape[0], remainder_len, -1)
            output_hidden_states.append(output_remainder)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)

        return output_hidden_states
    
class TitansSeqModelBlock(nn.Module):
    def __init__(self, config: TitansConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_mem_heads

        self.self_attn = None # Will change when implementing non-LMM variants.
        self.memory = TitansMemoryModule(config, layer_idx)

        # Additional details about projections in the Titans memory module layer is a bit confusing
        # between the paper and implementations. The paper mentions using a GSS (Mehta et al. 2023)
        # style of gating with additional gate proj and an output proj. TTT uses an output proj but leaves
        # mamba-style gating as optional. The lucidrains implementation does not use gating
        # and only optionally includes an output projection. The related papers ATLAS and MIRAS,
        # built on Titans by the same authors, uses gating but no output projection, only a post-norm
        # on the memory output (Fig 2. MIRAS, Fig 3. ATLAS). But GSS - the cited arch - doesn't have a post-norm,
        # it has a shared prenorm on the input! TTT has a post-norm but it's not optional regardless of whether you gate or not!
        # It is a chaotic fucking mess out there, to put it bluntly. There are many more inconsistencies I haven't even mentioned here.
        # The paper mentions gating, "normalization" (doesn't specify which), and an output projection.
        # We will implement all of these as options. By default, I will assume post-norm, gating, and output proj.
        # These can be turned off in the config separately if desired - it may not be worth the extra param cost.
        # Sorry for the crashout, code reader!
        self.use_gate = config.use_gate
        self.use_output_proj = config.use_output_proj
        self.out_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_output_proj:
            self.o_proj = nn.Linear(config.num_mem_heads * self.head_dim, config.hidden_size, bias=False)
        if self.use_gate:
            self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Persistent memory tokens.
        self.num_persistent_mem_tokens = config.num_persistent_mem_tokens
        if self.num_persistent_mem_tokens > 0:
            self.persistent_mem = nn.Parameter(
                torch.normal(
                    mean=0.0,
                    std=config.initializer_range,
                    size=(1, config.num_persistent_mem_tokens, config.hidden_size),
                )
            )
        else:
            self.persistent_mem = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: TitansCache | None = None
    ) -> torch.Tensor:
        # NOTE(TG): This forward function will get more complex when we implement non-LMM variants with attention.
        if self.use_gate:
            gate_values = F.silu(self.gate_proj(hidden_states))

        # Concatenate persistent memory tokens if applicable.
        if self.persistent_mem is not None:
            batch_size = hidden_states.size(0)
            persistent_mem_expanded = self.persistent_mem.expand(batch_size, -1, -1)
            hidden_states = torch.cat([persistent_mem_expanded, hidden_states], dim=1)

        hidden_states = self.memory(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache=cache
        )

        hidden_states = self.out_norm(hidden_states)

        # Remove persistent memory tokens after attn/memory module processing.
        if self.persistent_mem is not None:
            hidden_states = hidden_states[:, self.num_persistent_mem_tokens:, :]

        if self.use_gate:
            # Modulate memory output with gate values.
            hidden_states = gate_values * hidden_states
        if self.use_output_proj:
            hidden_states = self.o_proj(hidden_states)
        return hidden_states

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
    
class TitansDecoderLayer(nn.Module):
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
        self.head_dim = config.hidden_size // config.num_mem_heads

        # Persistent memory tokens.
        self.num_persistent_mem_tokens = config.num_persistent_mem_tokens
        if self.num_persistent_mem_tokens > 0:
            self.persistent_mem = nn.Parameter(
                torch.normal(
                    mean=0.0,
                    std=config.initializer_range,
                    size=(1, config.num_persistent_mem_tokens, config.hidden_size),
                )
            )
        else:
            self.persistent_mem = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: TitansCache | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        # Self-Attention block would go here for non-LMM variants.
        # hidden_states = self.self_attn(...)
        # Concatenate persistent memory tokens if applicable.
        if self.persistent_mem is not None:
            batch_size = hidden_states.size(0)
            persistent_mem_expanded = self.persistent_mem.expand(batch_size, -1, -1)
            hidden_states = torch.cat([persistent_mem_expanded, hidden_states], dim=1)

        hidden_states = self.memory(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache=cache
        )
        # Remove persistent memory tokens after attn/memory module processing.
        if self.persistent_mem is not None:
            hidden_states = hidden_states[:, self.num_persistent_mem_tokens:, :]
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
        self.num_persistent_mem_tokens = config.num_persistent_mem_tokens

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [TitansDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TitansRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

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

        hidden_states = inputs_embeds
        position_ids = torch.arange(0, hidden_states.size(1) + self.num_persistent_mem_tokens, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
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
