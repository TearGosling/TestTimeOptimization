import argparse
import logging
import os

import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from yaml import safe_load

from tto import * # Necessary to register AutoConfig/AutoModel mappings

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO(TG): Put repeat code into utils module.
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

def tokenize_function(examples, tokenizer, max_length=2048):
    out = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    out["labels"] = out.input_ids.clone()
    return out

def main():
    parser = argparse.ArgumentParser(description="Train a TTO model given a config.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = safe_load(f)

    model_args = config["model"]
    model_type = model_args.pop("type", "titans")

    training_args = config["training_args"]
    device = training_args.pop("device", "cuda")
    # resume_from_checkpoint is fed into Trainer separately
    resume_checkpoint = training_args.pop("resume_from_checkpoint", None)
    training_args = TrainingArguments(**training_args)

    LOG.info("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # If checkpoint has not been provided, initialize from config
    if resume_checkpoint is None:
        model_config_class = MODEL_CONFIG_MAPPING[model_type].config_class
        model_class = MODEL_CONFIG_MAPPING[model_type].model_class
        model_config = model_config_class(**model_args)
        model = model_class(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(resume_checkpoint)

    model = model.to(device=device)
    num_params = sum(p.numel() for p in model.parameters())
    LOG.info(f"Model initialized with {num_params:,} parameters.")

    # Tokenize dataset.
    dataset = load_dataset(config["dataset"]["name"], split="train")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config["dataset"]["max_seq_length"]},
        remove_columns=dataset.column_names,
        num_proc = config["dataset"].get("num_proc", os.cpu_count()),
    )

    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length=config["dataset"]["max_seq_length"],
        return_tensors="pt",
        #padding='max_length'
    )
    LOG.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    LOG.info("Training complete. Check your output directory for checkpoints.")

if __name__ == "__main__":
    main()
