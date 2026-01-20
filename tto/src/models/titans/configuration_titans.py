from transformers import PretrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import logging

logger = logging.get_logger(__name__)

class TitansConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TitansModel`]. It is used to instantiate a Titans
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of a 1B param model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Titans model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TitansModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        
    """
    model_type = "titans"

    def __init__(
        self,
        # General model settings
        vocab_size: int | None = 32768,
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
        **kwargs,
    ):
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
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
