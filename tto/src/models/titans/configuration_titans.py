from transformers import PretrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import logging

logger = logging.get_logger(__name__)

class TitansConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TitansModel`]. It is used to instantiate a Titans
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of a 1.5B param model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the Titans model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TitansModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5504):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Titans model.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions and memory states.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        conv_kernel (`int`, *optional*, defaults to 4):
            Kernel size for the convolutional layers.
        attention_conv (`bool`, *optional*, defaults to `False`):
            Whether to apply convolution to the QKV projections in the attention layers.
        rms_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply RMS normalization to the query and key projections in the memory layers. If `False`, L2 normalization is used.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Titans model. Distinct from `num_mem_heads`.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        sliding_window (`int`, *optional*, defaults to 2048):
            Sliding window attention window size if using the `"mag"` or `"mal"` variants. If not specified, will default to `2048`.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to `2048*64`):
            The maximum sequence length that this model might ever be used with. Note that this parameter applies only to rotary positional embeddings.
        variant (`str`, *optional*, defaults to `"lmm"`):
            The Titans variant to use. Four variants are supported: `"lmm"`(Titans memory module with no separate attention layer), `"mal"` (Memory as a Layer), `"mag"` (Memory as a Gate), and `"mac"` (Memory as a Context). Currently, only `"lmm"` is implemented.
        chunk_size (`int`, *optional*, defaults to 8):
            The chunk size (also known as "mini-batch size") to use for the memory layer updates.
        mem_expansion_factor (`float`, *optional*, defaults to 4.0):
            The expansion factor for the memory layer's MLP.
        num_mem_heads (`int`, *optional*, defaults to 32):
            Number of memory heads for each memory layer in the Titans model. Distinct from `num_attention_heads`.
        num_persistent_mem_tokens (`int`, *optional*, defaults to 4):
            Number of persistent memory tokens to use in each memory layer.
        use_output_proj (`bool`, *optional*, defaults to `True`):
            Whether to use an output projection layer in the memory layers.
        use_gate (`bool`, *optional*, defaults to `True`):
            Whether to use a gating mechanism in the memory layers.
    """
    model_type = "titans"

    def __init__(
        self,
        # General model settings
        vocab_size: int | None = 32000,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 5504,
        num_hidden_layers: int | None = 24,
        hidden_act: str | None = "silu",
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = False,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        conv_kernel: int | None = 4,
        attention_conv: bool = False,
        rms_qk_norm: bool = False,
        # Attention and pos embed settings
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        sliding_window: int | None = 2048,
        rope_parameters: RopeParameters | dict[str, RopeParameters] = None,
        attention_dropout: float | None = 0.0,
        max_position_embeddings: int | None = 2048 * 64,
        # Titans memory layer settings
        variant: str = "lmm",
        chunk_size: int = 8,
        mem_expansion_factor: float = 4.0,
        num_mem_heads: int = 32,
        num_persistent_mem_tokens: int = 4,
        use_output_proj: bool = True,
        use_gate: bool = True,
        scan_checkpoint_group_size: int = 8,
        **kwargs,
    ):
        if variant != "lmm":
            raise NotImplementedError(f"Currently only 'lmm' variant is supported, got {variant}.")
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.attention_conv = attention_conv
        self.conv_kernel = conv_kernel
        self.rms_qk_norm = rms_qk_norm

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.rope_parameters = rope_parameters
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        
        self.variant = variant
        self.chunk_size = chunk_size
        self.num_mem_heads = num_mem_heads
        self.num_persistent_mem_tokens = num_persistent_mem_tokens
        self.mem_expansion_factor = mem_expansion_factor
        self.use_output_proj = use_output_proj
        self.use_gate = use_gate
        self.scan_checkpoint_group_size = scan_checkpoint_group_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
