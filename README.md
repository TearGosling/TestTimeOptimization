# Test Time Training Models

Heavily WIP. May never be finished. Oh well!

An (attempted!) implementation of "test-time training" language models for long context in PyTorch, where an inner memory module learns to encode memory over the sequence by optimizing its weights as it processes tokens, introduced by [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) by Sun, et. al.

This likely won't work because I am not good at computer, but it's worth a shot anyway.

This framework will use the Hugging Face ecosystem, including the Transformers `Trainer`. Easier this way.

---

# Installation

The usual.

`pip3 install -r requirements.txt`

---

# Architecture Implementation Checklist

- [X] [**Learning to (Learn at Test Time): RNNs with Expressive Hidden States**](https://arxiv.org/abs/2407.04620) (copied from the [official implementation](https://github.com/test-time-training/ttt-lm-pytorch))
- [X] [**Titans: Learning to Memorize at Test Time**](https://arxiv.org/abs/2501.00663) (LMM variant only at the moment)
- [ ] [**Test-Time Training Done Right**](https://arxiv.org/abs/2505.23884)
- [ ] [**ATLAS: Learning to Optimally Memorize the Context at Test Time**](https://arxiv.org/abs/2505.23735)
- [ ] [**TNT: Improving Chunkwise Training For Test-Time Memorization**](https://arxiv.org/abs/2511.07343)
- [ ] [**ViT³: Unlocking Test-Time Training in Vision**](https://arxiv.org/abs/2512.01643)
- [ ] [**It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization**](https://arxiv.org/abs/2504.13173) (all variants)
- [ ] [**Nested Learning: The Illusion of Deep Learning Architecture**](https://arxiv.org/abs/2512.24695)

Will add more papers as I find them.

---

# Inference (basic)

An extremely basic inference script has been provided to test TTO models using Hugging Face's native `generate` function.
```
usage: infer.py [-h] [--model MODEL] [--config-type CONFIG_TYPE] [--tokenizer TOKENIZER] [--device DEVICE] [--dtype DTYPE] [--seed SEED] [--max_new_tokens MAX_NEW_TOKENS]
                [--temperature TEMPERATURE] [--top_k TOP_K] [--top_p TOP_P] [--min_p MIN_P] [--repetition_penalty REPETITION_PENALTY]

Inference script for TTO model (or any HF model in general, technically)

options:
  -h, --help            show this help message and exit
  --model MODEL         Pretrained model name or path. Cannot be supplied alongside --config-type.
  --config-type CONFIG_TYPE
                        Model config type (e.g., 'ttt' or 'titans') to initialize from scratch using default values. Cannot be supplied alongside --model.
  --tokenizer TOKENIZER
                        Pretrained tokenizer name or path. Defaults to 'NousResearch/Llama-2-7b-hf'
  --device DEVICE       Device to run the model on. Defaults to 'cuda:0'
  --dtype DTYPE         Data type for model weights (e.g., 'float16', 'bfloat16', 'float32'). Defaults to 'float32'
  --seed SEED           Random seed for reproducibility, if desired. Defaults to None (no fixed seed)
  --max_new_tokens MAX_NEW_TOKENS
                        Maximum number of new tokens to generate. Defaults to 50
  --temperature TEMPERATURE
                        Sampling temperature. Defaults to 1.0 (no scaling)
  --top_k TOP_K         Top-k sampling parameter. Defaults to None (no top-k filtering)
  --top_p TOP_P         Top-p sampling parameter. Defaults to None (no top-p filtering)
  --min_p MIN_P         Minimum probability for nucleus sampling. Defaults to None (no minimum probability)
  --repetition_penalty REPETITION_PENALTY
                        Repetition penalty for generation. Defaults to 1.0 (no penalty)
```

---

# Training (very unstable for Titans)

Use `train.py` to train a model using a config. The script is extremely basic for now; the Hugging Face `Trainer` class is used to train the models and the script supports single-GPU only.

```
python3 train.py <PATH_TO_CONFIG_FILE>
```

The config must be in a YAML file with the following structure:
```
dataset:
  name: [name of your dataset on Hugging Face]
  max_seq_length: [max sequence length of the data]
model:
  type: ["titans", "ttt"] - select your model.
  # Arguments supplied below must match keys found in the model configuration file.
  vocab_size: 32000
  hidden_size: 1024
  intermediate_size: 2736
  num_hidden_layers: 16
  # ...and so on.
tokenizer: [Path to HF Tokenizer]
training_args:
  device: [DEVICE]
  project: [W&B PROJECT NAME]
  seed: [SEED]
  # Most of the rest are direct feed-ins to Hugging Face `TrainingArguments`. Arguments not supplied fall back to their defaults.
  gradient_checkpointing: true
  learning_rate: [LEARNING_RATE]
  logging_steps: 1
  lr_scheduler_type: [LR SCHEDULER TYPE]
  max_grad_norm: [MAX GRAD NORM]
  per_device_train_batch_size: [BATCH SIZE]
  output_dir: "YOUR_OUTPUT_DIR"
  num_train_epochs: 1
  report_to: "wandb"
```

Training on Titans is currently **very unstable**, with high grad norms. I'm investigating the cause of this. An example config, `titans_lmm_300m.yml`, has been provided in the `configs` folder as an example. It will train on an RTX 3090, taking up roughly 18 GB of VRAM. I will work on getting that number to drop lower later.