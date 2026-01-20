"""Training script for Titans with HuggingFace dataset and wandb logging.

Usage:
    python -m tto.src.models.titans.train

Or from the titans directory:
    python train.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
import time

try:
    from .modeling_titans import TitansForCausalLM
    from .configuration_titans import TitansConfig
except ImportError:
    from modeling_titans import TitansForCausalLM
    from configuration_titans import TitansConfig

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Model architecture
    "model": {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_mem_heads": 4,
        "chunk_size": 32,
        "mem_expansion_factor": 4.0,
        "rms_norm_eps": 1e-6,
    },
    # Training
    "training": {
        "batch_size": 4,
        "max_length": 256,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "steps": 1000,
        "warmup_steps": 100,
    },
    # Logging
    "logging": {
        "log_every": 1,           # Log metrics every N steps
        "sample_every": 50,       # Generate samples every N steps
        "eval_every": 25,         # Run eval every N steps
        "num_eval_batches": 5,    # Number of batches for eval
    },
    # Data
    "data": {
        "dataset": "roneneldan/TinyStories",
        "tokenizer": "gpt2",
    },
    # Sampling
    "sampling": {
        "min_p": 0.05,
        "temperature": 0.8,
        "repetition_penalty": 1.1,
        "max_new_tokens": 50,
    },
    # wandb
    "wandb": {
        "project": "titans",
        "name": None,  # Auto-generated if None
        "enabled": True,
    },
}

# Fixed test prompts for tracking generation quality over time
TEST_PROMPTS = [
    "Once upon a time",
    "The little dog",
    "In a magical forest",
]


# =============================================================================
# UTILITIES
# =============================================================================
def min_p_sampling(logits: torch.Tensor, min_p: float = 0.05, temperature: float = 1.0) -> torch.Tensor:
    """Min-p sampling: keep tokens with prob >= min_p * max_prob."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    mask = probs >= (min_p * max_prob)
    masked_probs = probs * mask
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.multinomial(masked_probs, num_samples=1).squeeze(-1)


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.1) -> torch.Tensor:
    """Apply repetition penalty to logits."""
    for i in range(input_ids.shape[0]):
        unique_tokens = input_ids[i].unique()
        logits[i, unique_tokens] = torch.where(
            logits[i, unique_tokens] > 0,
            logits[i, unique_tokens] / penalty,
            logits[i, unique_tokens] * penalty,
        )
    return logits


def reset_memory_state(model):
    """Reset memory state for all layers."""
    for layer in model.model.layers:
        layer.memory.params_dict = None


@torch.no_grad()
def generate_sample(
    model: TitansForCausalLM,
    tokenizer,
    prompt: str,
    config: dict,
    device: torch.device,
) -> str:
    """Generate text from prompt."""
    model.eval()
    reset_memory_state(model)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(config["sampling"]["max_new_tokens"]):
        outputs = model(input_ids=generated)
        next_logits = outputs.logits[:, -1, :]
        next_logits = apply_repetition_penalty(
            next_logits, generated, config["sampling"]["repetition_penalty"]
        )
        next_token = min_p_sampling(
            next_logits,
            min_p=config["sampling"]["min_p"],
            temperature=config["sampling"]["temperature"],
        )
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    model.train()
    return tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.no_grad()
def evaluate(model, eval_dataloader, device, num_batches):
    """Run evaluation and return metrics."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(eval_dataloader):
        if i >= num_batches:
            break

        reset_memory_state(model)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, labels=labels)
        num_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return {"eval_loss": avg_loss, "eval_ppl": torch.exp(torch.tensor(avg_loss)).item()}


def collate_fn(batch, tokenizer, max_length):
    """Collate function for DataLoader."""
    texts = [item["text"] for item in batch]
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encodings


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    config = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize wandb
    if config["wandb"]["enabled"]:
        run_name = config["wandb"]["name"] or f"titans-{config['model']['hidden_size']}d-{config['model']['num_hidden_layers']}L"
        wandb.init(
            project=config["wandb"]["project"],
            name=run_name,
            config=config,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model config
    model_config = TitansConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config["model"]["hidden_size"],
        intermediate_size=config["model"]["intermediate_size"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        num_mem_heads=config["model"]["num_mem_heads"],
        chunk_size=config["model"]["chunk_size"],
        mem_expansion_factor=config["model"]["mem_expansion_factor"],
        rms_norm_eps=config["model"]["rms_norm_eps"],
        max_position_embeddings=config["training"]["max_length"],
    )

    model = TitansForCausalLM(model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params/1e6:.2f}M parameters")

    if config["wandb"]["enabled"]:
        wandb.log({"model/num_params": num_params})

    # Load dataset
    print(f"Loading dataset: {config['data']['dataset']}")
    train_dataset = load_dataset(config["data"]["dataset"], split="train", streaming=True)
    eval_dataset = load_dataset(config["data"]["dataset"], split="validation", streaming=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=lambda b: collate_fn(b, tokenizer, config["training"]["max_length"]),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=lambda b: collate_fn(b, tokenizer, config["training"]["max_length"]),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    def get_lr(step):
        warmup = config["training"]["warmup_steps"]
        if step < warmup:
            return step / warmup
        return 1.0

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    pbar = tqdm(range(config["training"]["steps"]), desc="Training")
    step_times = []

    for step in pbar:
        step_start = time.time()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        labels[attention_mask == 0] = -100

        reset_memory_state(model)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["grad_clip"]
        )

        lr_scale = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = config["training"]["lr"] * lr_scale

        optimizer.step()

        step_time = time.time() - step_start
        step_times.append(step_time)

        num_tokens = (labels != -100).sum().item()
        tokens_per_sec = num_tokens / step_time

        metrics = {
            "train/loss": loss.item(),
            "train/ppl": torch.exp(loss).item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": config["training"]["lr"] * lr_scale,
            "train/tokens_per_sec": tokens_per_sec,
            "train/step_time": step_time,
        }

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "ppl": f"{torch.exp(loss).item():.1f}",
            "tok/s": f"{tokens_per_sec:.0f}",
        })

        if config["wandb"]["enabled"] and (step + 1) % config["logging"]["log_every"] == 0:
            wandb.log(metrics, step=step)

        if (step + 1) % config["logging"]["eval_every"] == 0:
            try:
                eval_metrics = evaluate(
                    model, eval_loader, device, config["logging"]["num_eval_batches"]
                )
            except StopIteration:
                eval_iter = iter(eval_loader)
                eval_metrics = evaluate(
                    model, eval_loader, device, config["logging"]["num_eval_batches"]
                )

            print(f"\n[Step {step+1}] Eval loss: {eval_metrics['eval_loss']:.4f}, Eval PPL: {eval_metrics['eval_ppl']:.2f}")

            if config["wandb"]["enabled"]:
                wandb.log(eval_metrics, step=step)

        if (step + 1) % config["logging"]["sample_every"] == 0:
            print(f"\n{'='*60}")
            print(f"Step {step + 1} - Sample generations:")
            print(f"{'='*60}")

            generation_table = []
            for prompt in TEST_PROMPTS:
                generated = generate_sample(model, tokenizer, prompt, config, device)
                print(f"Prompt: {prompt!r}")
                print(f"Output: {generated!r}")
                print("-" * 40)
                generation_table.append([prompt, generated])

            if config["wandb"]["enabled"]:
                table = wandb.Table(columns=["prompt", "generation"], data=generation_table)
                wandb.log({f"generations/step_{step+1}": table}, step=step)

            print(f"{'='*60}\n")

    avg_step_time = sum(step_times) / len(step_times)
    print(f"\nTraining complete!")
    print(f"Average step time: {avg_step_time:.3f}s")
    print(f"Average tokens/sec: {num_tokens / avg_step_time:.0f}")

    if config["wandb"]["enabled"]:
        wandb.finish()


if __name__ == "__main__":
    main()
