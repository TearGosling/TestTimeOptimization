# Copyright 2026 The Titans authors and Hugging Face contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Titans model configuration."""

from __future__ import annotations

from transformers.configuration_utils import PreTrainedConfig


class TitansConfig(PreTrainedConfig):
    r"""
    Configuration class for [`TitansModel`] and [`TitansForCausalLM`].

    The configuration exposes the long-term neural memory module from
    "Titans: Learning to Memorize at Test Time" and the four supported
    architecture variants:

    - `"lmm"`: memory-only Long-term Memory Module sequence model.
    - `"mac"`: Memory as a Context.
    - `"mag"`: Memory as a Gate.
    - `"mal"`: Memory as a Layer.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size.
        hidden_size (`int`, *optional*, defaults to 768):
            Decoder hidden size.
        intermediate_size (`int`, *optional*, defaults to 2048):
            SwiGLU feed-forward intermediate size.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of decoder blocks.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention query heads.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads for grouped-query attention. Defaults to
            `num_attention_heads`.
        variant (`str`, *optional*, defaults to `"mac"`):
            Titans variant. One of `"lmm"`, `"mac"`, `"mag"`, or `"mal"`.
        memory_num_heads (`int`, *optional*):
            Number of independent memory heads. Defaults to
            `num_attention_heads`.
        memory_head_dim (`int`, *optional*):
            Per-memory-head dimension. Defaults to
            `hidden_size // memory_num_heads`.
        memory_num_layers (`int`, *optional*, defaults to 2):
            Number of layers in each per-head neural memory MLP. A value of 1
            gives a linear associative memory.
        memory_hidden_size (`int`, *optional*):
            Hidden width of the memory MLP. Defaults to
            `memory_mlp_expansion * memory_head_dim`.
        memory_mlp_expansion (`int`, *optional*, defaults to 4):
            Expansion factor used when `memory_hidden_size` is not supplied.
        memory_chunk_size (`int`, *optional*, defaults to 16):
            Chunk size used by the paper's parallel inner-loop training rule.
        memory_residual (`bool`, *optional*, defaults to `True`):
            Whether to use a residual connection in the memory MLP.
        mac_segment_size (`int`, *optional*):
            Segment size used by Memory as a Context. Defaults to
            `memory_chunk_size`.
        persistent_memory_tokens (`int`, *optional*, defaults to 4):
            Number of learnable input-independent persistent memory tokens.
        sliding_window (`int`, *optional*, defaults to 256):
            Sliding-window size used by MAG and MAL attention. Use `None` for
            full causal attention.
        memory_theta_scale (`float`, *optional*, defaults to 1.0):
            Scale for the data-dependent inner-loop learning rate theta.
        memory_eta_scale (`float`, *optional*, defaults to 1.0):
            Scale for the data-dependent surprise decay eta.
        memory_alpha_scale (`float`, *optional*, defaults to 1.0):
            Scale for the data-dependent memory decay alpha.
        memory_theta_bias (`float`, *optional*, defaults to -2.0):
            Initial learnable logit bias for the inner-loop learning rate theta.
        memory_eta_bias (`float`, *optional*, defaults to 2.0):
            Initial learnable logit bias for the surprise decay eta.
        memory_alpha_bias (`float`, *optional*, defaults to -5.0):
            Initial learnable logit bias for the memory decay alpha.
    """

    model_type = "titans"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = 2,
        tie_word_embeddings: bool = True,
        pretraining_tp: int = 1,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        variant: str = "mac",
        memory_num_heads: int | None = None,
        memory_head_dim: int | None = None,
        memory_num_layers: int = 2,
        memory_hidden_size: int | None = None,
        memory_mlp_expansion: int = 4,
        memory_activation: str = "silu",
        memory_chunk_size: int = 16,
        memory_residual: bool = True,
        mac_segment_size: int | None = None,
        persistent_memory_tokens: int = 4,
        sliding_window: int | None = 256,
        qkv_conv_kernel: int = 4,
        attention_qkv_conv_kernel: int | None = None,
        memory_qkv_conv_kernel: int | None = None,
        use_attention_convolution: bool = True,
        use_memory_convolution: bool = True,
        memory_theta_scale: float = 1.0,
        memory_eta_scale: float = 1.0,
        memory_alpha_scale: float = 1.0,
        memory_theta_bias: float = -2.0,
        memory_eta_bias: float = 2.0,
        memory_alpha_bias: float = -5.0,
        memory_loss_scale: float | None = None,
        l2_norm_eps: float = 1e-6,
        use_parallel_memory_training: bool = True,
        parallel_scan_epsilon: float = 1e-6,
        gate_bias: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.resid_dropout = resid_dropout

        self.variant = variant.lower().strip()
        self.memory_num_heads = num_attention_heads if memory_num_heads is None else memory_num_heads
        self.memory_head_dim = hidden_size // self.memory_num_heads if memory_head_dim is None else memory_head_dim
        self.memory_num_layers = memory_num_layers
        self.memory_mlp_expansion = memory_mlp_expansion
        self.memory_hidden_size = (
            self.memory_head_dim * memory_mlp_expansion if memory_hidden_size is None else memory_hidden_size
        )
        self.memory_activation = memory_activation
        self.memory_chunk_size = memory_chunk_size
        self.memory_residual = memory_residual
        self.mac_segment_size = memory_chunk_size if mac_segment_size is None else mac_segment_size
        self.persistent_memory_tokens = persistent_memory_tokens
        self.sliding_window = sliding_window

        self.qkv_conv_kernel = qkv_conv_kernel
        self.attention_qkv_conv_kernel = (
            qkv_conv_kernel if attention_qkv_conv_kernel is None else attention_qkv_conv_kernel
        )
        self.memory_qkv_conv_kernel = qkv_conv_kernel if memory_qkv_conv_kernel is None else memory_qkv_conv_kernel
        self.use_attention_convolution = use_attention_convolution
        self.use_memory_convolution = use_memory_convolution

        self.memory_theta_scale = memory_theta_scale
        self.memory_eta_scale = memory_eta_scale
        self.memory_alpha_scale = memory_alpha_scale
        self.memory_theta_bias = memory_theta_bias
        self.memory_eta_bias = memory_eta_bias
        self.memory_alpha_bias = memory_alpha_bias
        self.memory_loss_scale = 1.0 / self.memory_head_dim if memory_loss_scale is None else memory_loss_scale
        self.l2_norm_eps = l2_norm_eps
        self.use_parallel_memory_training = use_parallel_memory_training
        self.parallel_scan_epsilon = parallel_scan_epsilon
        self.gate_bias = gate_bias

        self._validate()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate(self) -> None:
        variants = {"lmm", "mac", "mag", "mal"}
        if self.variant not in variants:
            raise ValueError(f"`variant` must be one of {sorted(variants)}, got {self.variant!r}.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads`.")
        if self.memory_num_heads <= 0 or self.memory_head_dim <= 0:
            raise ValueError("`memory_num_heads` and `memory_head_dim` must be positive.")
        if self.memory_num_layers < 1:
            raise ValueError("`memory_num_layers` must be at least 1.")
        if self.memory_chunk_size < 1:
            raise ValueError("`memory_chunk_size` must be at least 1.")
        if self.mac_segment_size < 1:
            raise ValueError("`mac_segment_size` must be at least 1.")
        if self.persistent_memory_tokens < 0:
            raise ValueError("`persistent_memory_tokens` cannot be negative.")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("`sliding_window` must be positive or `None`.")
        if self.attention_qkv_conv_kernel < 1 or self.memory_qkv_conv_kernel < 1:
            raise ValueError("Convolution kernels must be at least 1.")


__all__ = ["TitansConfig"]
