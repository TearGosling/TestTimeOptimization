import argparse
import logging
import os

import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    Trainer,
    TrainingArguments,
    set_seed,
)
from yaml import safe_load

from tto import LigerKernelConfig, MODEL_CONFIG_MAPPING, patch_model_with_liger
from tto.src.utils import configure_distributed_environment

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
        padding="max_length",
        max_length=max_length,
    )
    # Ignore padding positions in LM loss.
    out["labels"] = [
        [token_id if mask == 1 else -100 for token_id, mask in zip(ids, masks)]
        for ids, masks in zip(out["input_ids"], out["attention_mask"])
    ]
    return out

def main():
    parser = argparse.ArgumentParser(description="Train a TTO model given a config.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = safe_load(f)
    liger_config = LigerKernelConfig(**config.get("liger_kernel", {}))

    model_args = config["model"]
    model_type = model_args.pop("type", "titans")

    training_args = config["training_args"]
    if "device" in training_args:
        raise ValueError(
            "Remove training_args.device. Trainer selects each rank's device; "
            "use CUDA_VISIBLE_DEVICES/the launcher for GPUs or use_cpu: true for CPU."
        )
    # resume_from_checkpoint is fed into Trainer separately
    resume_checkpoint = training_args.pop("resume_from_checkpoint", None)
    if training_args.get("use_liger_kernel"):
        raise ValueError("Configure Liger components in the top-level 'liger_kernel' section instead of use_liger_kernel.")
    configure_distributed_environment(use_cpu=training_args.get("use_cpu", False))
    training_args = TrainingArguments(**training_args)
    LOG.setLevel(training_args.get_process_log_level())
    LOG.info(
        "Training rank %s/%s on %s (local rank %s).",
        training_args.process_index, training_args.world_size,
        training_args.device, training_args.local_process_index,
    )
    set_seed(training_args.seed)
    # Backup W&B project naming since HF Trainer likes putting everything under the 'huggingface' project
    os.environ["WANDB_PROJECT"] = training_args.project

    LOG.info("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # If checkpoint has not been provided, initialize from config
    if resume_checkpoint is None:
        model_config_class = MODEL_CONFIG_MAPPING[model_type].config_class
        model_config = model_config_class(**model_args)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        # Attention backends are runtime options, not serialized architecture.
        model = AutoModelForCausalLM.from_pretrained(
            resume_checkpoint, attn_implementation=model_args.get("attn_implementation")
        )

    model = patch_model_with_liger(model, liger_config)
    num_params = sum(p.numel() for p in model.parameters())
    LOG.info(f"Model initialized with {num_params:,} parameters.")

    # The leader populates the datasets cache before other ranks read it.
    dataset_args = config["dataset"]
    shared_cache = dataset_args.get("shared_cache", True)
    if not isinstance(shared_cache, bool):
        raise TypeError("dataset.shared_cache must be a boolean.")
    with training_args.main_process_first(local=not shared_cache, desc="dataset preprocessing"):
        dataset = load_dataset(dataset_args["name"], split="train")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "max_length": dataset_args["max_seq_length"]},
            remove_columns=dataset.column_names,
            num_proc=dataset_args.get("num_proc"),
        )

    collate_fn = default_data_collator
    LOG.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    finally:
        trainer.accelerator.end_training()
    LOG.info("Training complete. Check your output directory for checkpoints.")

if __name__ == "__main__":
    main()
