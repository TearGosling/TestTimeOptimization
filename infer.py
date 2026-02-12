import argparse
import logging
import random
import time

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tto import *

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    TYPE_MAP = {
        ("float16", "fp16"): torch.float16,
        ("bfloat16", "bf16"): torch.bfloat16,
        ("float32", "fp32"): torch.float32,
    }
    for keys, dtype in TYPE_MAP.items():
        if dtype_str in keys:
            return dtype
    raise ValueError(f"Unsupported dtype string: {dtype_str}")

def build_gen_config(args: argparse.Namespace) -> GenerationConfig:
    do_sample = args.temperature > 0.0
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=do_sample,
        max_new_tokens=args.max_new_tokens,
    )
    return gen_config

def parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for TTO model (or any HF model in general, technically)")
    # Users can either provide a model name or path, or a config type to initialize from scratch
    parser.add_argument("--model", type=str, default=None, help="Pretrained model name or path. Cannot be supplied alongside --config-type.")
    parser.add_argument("--config-type", type=str, default=None, help="Model config type (e.g., 'ttt' or 'titans') to initialize from scratch using default values. Cannot be supplied alongside --model.")
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-hf", help="Pretrained tokenizer name or path. Defaults to 'NousResearch/Llama-2-7b-hf'")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on. Defaults to 'cuda:0'")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for model weights (e.g., 'float16', 'bfloat16', 'float32'). Defaults to 'float32'")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility, if desired. Defaults to None (no fixed seed)")
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate. Defaults to 50")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Defaults to 1.0 (no scaling)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter. Defaults to None (no top-k filtering)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter. Defaults to None (no top-p filtering)")
    parser.add_argument("--min_p", type=float, default=None, help="Minimum probability for nucleus sampling. Defaults to None (no minimum probability)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation. Defaults to 1.0 (no penalty)")
    args = parser.parse_args()

    assert (args.model is not None) != (args.config_type is not None), "Either --model or --config-type must be provided, but not both."
    return args

def main():
    args = parse_args_from_argv()
    gen_config = build_gen_config(args)

    prompt = input("Enter your prompt: ")
    dtype = get_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_load_time = time.time()
    if args.seed is not None:
        LOG.info(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Model loading
    if args.config_type is not None:
        LOG.info(f"Initializing model from scratch with config type: {args.config_type}")
        if args.config_type.lower() == "ttt":
            configuration = TTTConfig()
            model = TTTForCausalLM(configuration)
        elif args.config_type.lower() == "titans":
            configuration = TitansConfig()
            model = TitansForCausalLM._from_config(configuration)
        else:
            raise ValueError(f"Unsupported config type: {args.config_type}")
    else:
        LOG.info(f"Loading pretrained model from: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval().to(device=args.device, dtype=dtype)
    model_load_time = time.time() - model_load_time
    LOG.info(f"Model loaded in {model_load_time:.2f} seconds")

    num_params = sum(p.numel() for p in model.parameters())
    LOG.info(f"Number of model parameters: {num_params:,}")

    LOG.info("Starting generation...")
    # Tokenize prompt and feed in
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device=args.device)
    gen_time = time.time()
    out_ids = model.generate(input_ids=input_ids, generation_config=gen_config)
    gen_time = time.time() - gen_time

    tokens_per_sec = (out_ids.size(1) - input_ids.size(1)) / gen_time
    LOG.info(f"Generation time: {gen_time:.2f} seconds ({tokens_per_sec:.2f} tokens/sec)")

    out_str = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    print("Generated Output:")
    print(out_str)

if __name__ == "__main__":
    main()
