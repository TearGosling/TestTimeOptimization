from collections import namedtuple

from transformers import AutoConfig, AutoModelForCausalLM

from .src.models.titans import TitansConfig, TitansForCausalLM
from .src.models.ttt import TTTConfig, TTTForCausalLM
from .src.utils import LigerKernelConfig, patch_model_with_liger

ModelConfigPair = namedtuple("ModelConfigPair", ["config_class", "model_class"])

# Implement our custom models into the AutoModel system.
AutoConfig.register("ttt", TTTConfig)
AutoModelForCausalLM.register(TTTConfig, TTTForCausalLM)
AutoConfig.register("titans", TitansConfig)
AutoModelForCausalLM.register(TitansConfig, TitansForCausalLM)

MODEL_CONFIG_MAPPING: dict[str, ModelConfigPair] = {
    "ttt": ModelConfigPair(TTTConfig, TTTForCausalLM),
    "titans": ModelConfigPair(TitansConfig, TitansForCausalLM),
}

__all__ = [
    "TTTConfig",
    "TTTForCausalLM",
    "TitansConfig",
    "TitansForCausalLM",
    "MODEL_CONFIG_MAPPING",
    "LigerKernelConfig",
    "patch_model_with_liger",
]
