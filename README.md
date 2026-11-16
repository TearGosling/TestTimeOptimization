# Test-Time Optimization

Research implementations of language models that learn from their context while
processing it, built with PyTorch and Hugging Face Transformers. The repository
currently includes Test-Time Training (TTT) and Titans, with more architectures
tracked in the implementation checklist below.

Work in progress. The training and inference scripts are experimental, and
passing tests does not establish reproduction of the papers' published results.

**NOTE**: Much of this repository has been vibe coded by GPT-6.1 Sol. There may be unexpected issues. Please feel free to open an issue or a PR if you see any problems.

## Contents

- [What is test-time optimization?](#what-is-test-time-optimization)
- [Supported models](#supported-models)
- [Installation](#installation)
- [Using the Python API](#using-the-python-api)
- [Training](#training)
- [Inference](#inference)
- [Attention backends](#attention-backends)
- [Tests and continuous integration](#tests-and-continuous-integration)
- [Repository layout](#repository-layout)
- [Architecture implementation checklist](#architecture-implementation-checklist)
- [TODOs](#todos-agenda)

## What is test-time optimization?

In the language models explored here, part of the model's state is a small
learner whose parameters change as it reads a sequence. An inner learning
objective turns incoming context into updates to that state. Later tokens can
retrieve information encoded by earlier updates.

There are two learning processes:

- During ordinary model training, an outer optimizer learns the language model
  and the parameters that determine how its adaptive state learns.
- During sequence processing, including inference, an inner update changes the
  adaptive memory using the current context. This usually updates a small
  per-sequence memory rather than fine-tuning the entire language model.

[TTT](https://arxiv.org/abs/2407.04620) treats the recurrent hidden state as a
learnable model and uses self-supervised updates to encode context.
[Titans](https://arxiv.org/abs/2501.00663) develops a neural long-term memory with
momentum and forgetting, and explores several ways to combine it with attention.
The exact update objective and memory architecture depend on the model.

These inner updates are part of the model's forward computation. They can run at
inference without a separate dataset or a call to the outer training optimizer.
The learned model weights and its changing sequence state have different roles.
A generation cache carries sequence state between calls without making those
updates permanent changes to the pretrained checkpoint.

The focus here is gradient-based memory and adaptation during sequence
processing. Other uses of the term "test-time optimization", such as searching
over multiple generated answers, are outside the current implementations.

## Supported models

| Model type | Configuration | Available choices |
| --- | --- | --- |
| `ttt` | [`TTTConfig`](tto/src/models/ttt/configuration_ttt.py) | `ttt_layer_type="linear"` or `"mlp"` |
| `titans` | [`TitansConfig`](tto/src/models/titans/configuration_titans.py) | `variant="lmm"`, `"mac"`, `"mag"`, or `"mal"` |

Titans variants are the memory-only Long-term Memory Module (LMM), Memory as a
Context (MAC), Memory as a Gate (MAG), and Memory as a Layer (MAL).

Importing `tto` registers both model families with Hugging Face's `AutoConfig`
and `AutoModelForCausalLM`. Model-specific options are documented in the
configuration files linked above. The full research roadmap remains in the
architecture checklist.

## Installation

### Requirements

- Python 3.11 or newer. Python 3.12 is used in CI.
- PyTorch 2.6 or newer and Transformers 5.23 or newer.
- Linux is the tested environment for the current dependency set.
- A compatible GPU/PyTorch build for CUDA training or FlashAttention. The
  standard regression tests can run on CPU.

The tested GPU environment uses Python 3.12, PyTorch 2.10.0 with CUDA 12.8, and
Transformers 5.23.0 on an RTX 3090. Hardware requirements vary with the model
dimensions, sequence length, and training configuration.

### 1. Clone the repository and create an environment

```bash
git clone https://github.com/TearGosling/TestTimeOptimization.git
cd TestTimeOptimization
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If the repository is already cloned, start with the environment commands.
Activate `.venv` again in each new terminal. The project is imported directly
from the checkout, so run the examples from the repository root.

### 2. Install PyTorch for your hardware

For GPU work, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
to choose a build compatible with your platform and driver before installing
the remaining dependencies.

For CPU-only development, this is the build used by CI:

```bash
python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
```

Choose the GPU or CPU installation appropriate to your machine. A CPU-only
PyTorch build cannot run CUDA kernels.

### 3. Install the project dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The requirements include Accelerate, Datasets, and Weights & Biases for the
training script. The `transformers[kernels]` extra provides the supported Hub
kernel loader. Installing Liger Kernel does not enable Liger integration in the
models automatically.

Check that the package imports and that PyTorch can see your GPU if needed:

```bash
python -c "from tto import TitansConfig, TTTConfig; print(TitansConfig.model_type, TTTConfig.model_type)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 4. Authenticate when needed

The example configuration uses the `NousResearch/Llama-2-7b-hf` tokenizer.
If Hugging Face requires access approval or authentication for a resource,
obtain access first and then sign in:

```bash
hf auth login
```

For runs with `report_to: "wandb"`, authenticate with:

```bash
wandb login
```

Set `training_args.report_to: "none"` to disable experiment reporting.
Authentication is not required for the local regression tests.

## Using the Python API

This small CPU example constructs a randomly initialized model and checks its
output shape without downloading a tokenizer or checkpoint:

```python
import torch
from transformers import AutoModelForCausalLM
from tto import TitansConfig

config = TitansConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    memory_num_heads=4,
    memory_head_dim=16,
    memory_hidden_size=32,
    memory_chunk_size=8,
    persistent_memory_tokens=0,
    variant="lmm",
)
model = AutoModelForCausalLM.from_config(config).eval()
input_ids = torch.tensor([[4, 5, 6, 7]])
with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)
print(outputs.logits.shape)  # torch.Size([1, 4, 128])
```

Use `TTTConfig` with `AutoModelForCausalLM.from_config` to construct a TTT model.
Models created from configuration have random weights. Load a trained
checkpoint before expecting meaningful generated text.

## Training

The training entry point accepts a YAML configuration. Start with
[`configs/titans_lmm_120m.yml`](configs/titans_lmm_120m.yml), which defines a
Titans LMM run with a 32K-token vocabulary.

```bash
python train.py configs/titans_lmm_120m.yml
```

For a distributed launch, configure Accelerate for your environment:

```bash
accelerate config
accelerate launch train.py configs/titans_lmm_120m.yml
```

### Configuration

| Section | Purpose |
| --- | --- |
| `dataset` | Hugging Face dataset `name`, `max_seq_length`, and optional tokenization `num_proc` |
| `model` | `type: "titans"` or `"ttt"`, followed by options accepted by that model's configuration class |
| `tokenizer` | A Hugging Face tokenizer ID or local tokenizer directory |
| `training_args` | Hugging Face `TrainingArguments`, plus the script's `device` and `resume_from_checkpoint` options |

The script loads the dataset's `train` split and expects a `text` column. Each
example is truncated or padded to `dataset.max_seq_length`, with padding ignored
in the language-model loss. The script does not pack multiple examples together.
Choose a tokenizer whose vocabulary matches the model's `vocab_size`.

Before starting a run, check the model dimensions and `output_dir`. The example
reports to Weights & Biases by default. Use `activation_checkpointing: true` to
enable activation checkpointing. Supported hardware can use `bf16: true` or
`fp16: true` in `training_args` for mixed-precision training.

Larger memory chunks do not always reduce VRAM, and changing their size can
change the learning behavior. Measure memory with your actual configuration
before increasing batch size or context length.

### Checkpoints and resuming

The example writes checkpoints under `./checkpoints/titans_lmm_120m`.
To resume, set the checkpoint directory in the YAML file:

```yaml
training_args:
  resume_from_checkpoint: "./checkpoints/titans_lmm_120m/checkpoint-1000"
```

Use a directory that exists from your run. The script loads the model from that
checkpoint and asks `Trainer` to resume its saved training state. The checkpoint
determines the model architecture when resuming.

## Inference

`infer.py` loads a checkpoint, asks for a prompt in the terminal, and calls the
model's Hugging Face `generate` method. For example, after producing the
checkpoint shown below:

```bash
python infer.py \
  --model ./checkpoints/titans_lmm_120m/checkpoint-1000 \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --device cuda:0 \
  --dtype bfloat16 \
  --max_new_tokens 100 \
  --temperature 0
```

Use the tokenizer associated with the training run. `--temperature 0` selects
greedy decoding in this script. CPU inference is available with `--device cpu`
and `--dtype float32`, though large models may be slow.

`--config-type titans` or `--config-type ttt` creates an untrained model using
that architecture's default configuration. It cannot be combined with `--model`
and is only useful for smoke checks. Default configurations can be large.

See the available arguments with:

```bash
python train.py --help
python infer.py --help
```

## Attention backends

Titans supports `attn_implementation="eager"`, `"sdpa"`, and
`"flash_attention_2"` through its configuration or Python API. Backend switching
is also available through `model.set_attn_implementation(...)`. This interface
is specific to the Titans implementation, not the TTT model.

FlashAttention requires compatible hardware and FP16/BF16 execution or autocast.
Transformers can fetch an official Hub kernel when the compiled `flash-attn`
package is absent, which requires network access on first use.

Titans MAL and MAG without persistent tokens can use native FlashAttention for
sliding-window attention. MAG's persistent-prefix/window pattern and MAC's
query-dependent prefix mask explicitly fall back to SDPA. That path preserves
the mask but still allocates a dense attention mask. Requesting attention
probabilities with `output_attentions=True` uses eager attention.

## Tests and continuous integration

Run the focused regression suite from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_titans_*.py' -v
```

The current suite targets Titans. It checks the memory updates against explicit
reference implementations, including outer gradients, and exercises causality,
cached generation, checkpoint loading, and attention backends. The standard
command runs on CPU without model downloads. The real FlashAttention GPU test
is skipped unless explicitly enabled:

```bash
TITANS_TEST_FLASH_ATTENTION=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_titans_attention.py' -v
```

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the CPU suite
on pushes and pull requests using Python 3.12. Commit and push the workflow
with the tests and model changes, then open **Actions**, select **Titans tests**,
and open a run to inspect the logs.

Once the workflow is on the default branch, **Run workflow** starts it manually.
The workflow does not configure branch protection. Require the **Titans CPU
tests** status check in repository rules if you want failed checks to block
updates to a protected branch. No PyPI publication or custom secrets are needed.

## Repository layout

```text
tto/
  __init__.py                 Hugging Face model registration
  src/models/
    ttt/                      TTT configuration and implementation
    titans/                   Titans configuration and implementation
configs/                      Example training configurations
tests/                        Titans regression tests
.github/workflows/tests.yml   CPU test workflow
train.py                      YAML-driven training entry point
infer.py                      Interactive generation entry point
requirements.txt              Project dependencies
```

## License

See [LICENSE](LICENSE) for the Apache License 2.0. The TTT implementation is
adapted from the official implementation linked in the architecture checklist.

---

# Architecture Implementation Checklist

- [X] [**Learning to (Learn at Test Time): RNNs with Expressive Hidden States**](https://arxiv.org/abs/2407.04620) (copied from the [official implementation](https://github.com/test-time-training/ttt-lm-pytorch))
- [X] [**Titans: Learning to Memorize at Test Time**](https://arxiv.org/abs/2501.00663) (all variants)
- [ ] [**Test-Time Training Done Right**](https://arxiv.org/abs/2505.23884)
- [ ] [**ATLAS: Learning to Optimally Memorize the Context at Test Time**](https://arxiv.org/abs/2505.23735)
- [ ] [**TNT: Improving Chunkwise Training For Test-Time Memorization**](https://arxiv.org/abs/2511.07343)
- [ ] [**ViT³: Unlocking Test-Time Training in Vision**](https://arxiv.org/abs/2512.01643)
- [ ] [**It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization**](https://arxiv.org/abs/2504.13173) (all variants)
- [ ] [**Nested Learning: The Illusion of Deep Learning Architecture**](https://arxiv.org/abs/2512.24695)

Will add more papers as I find them.

---

# TODOs (Agenda)

Various other things besides architectures themselves I have to implement, eventually (in no particular order):

- [ ] [Liger Kernel](https://github.com/linkedin/Liger-Kernel) support.
- [ ] Add proper parallel associative scan implementation
- [ ] Reduce redundancy by creating a common utils file
- [ ] Add custom Triton kernels
- [ ] Production-ready inference and training scripts
- [ ] Customizable model architectures
- [ ] Find good hyperparameters for example runs