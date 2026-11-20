"""Offline train.py regressions using real Trainer, Gloo and local checkpoints.

Run with:
    .venv/bin/python -m unittest discover -s tests -p test_training_distributed.py -v
"""

import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from datasets import Dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizerFast
import yaml


_ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT = 240
_SEED = 2501
_LENGTHS = (2, 7, 3, 8, 4, 6, 2, 5)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _terminate_groups(processes):
    # torchrun and its workers must not survive a failed assertion or timeout.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.wait(timeout=10)


@unittest.skipUnless(
    os.name == "posix" and dist.is_available() and dist.is_gloo_available(),
    "Requires POSIX process groups and torch.distributed Gloo",
)
class TrainingDistributedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tto-trainer-ddp-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = self.directory / "datasets-cache"
        self.dataset = self.directory / "dataset"
        self.dataset.mkdir()
        self.tokenizer_path = self.directory / "tokenizer"
        vocabulary = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
        vocabulary.update({f"word{index}": index + 4 for index in range(28)})
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            model_max_length=8,
        )
        tokenizer.save_pretrained(self.tokenizer_path)
        self.expected_ids = [
            [4 + (row * 3 + column) % 28 for column in range(length)]
            for row, length in enumerate(_LENGTHS)
        ]
        with (self.dataset / "train.jsonl").open("w") as stream:
            for ids in self.expected_ids:
                stream.write(json.dumps({"text": " ".join(f"word{token - 4}" for token in ids)}) + "\n")

        # Do not inherit launchers, rendezvous settings, GPU visibility or caches.
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SLURM_", "TORCHELASTIC_", "ACCELERATE_", "HF_", "WANDB_", "PMI_", "PMIX_", "OMPI_"))
            and key not in {
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
            }
        }
        self.environment.update({
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(self.directory / "hf-home"),
            "HF_DATASETS_CACHE": str(self.cache),
            "HF_HUB_CACHE": str(self.directory / "hub-cache"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "XDG_CACHE_HOME": str(self.directory / "xdg-cache"),
            "TORCH_HOME": str(self.directory / "torch-cache"),
            "WANDB_MODE": "disabled",
            "PYTHONHASHSEED": str(_SEED),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "GLOO_SOCKET_IFNAME": "lo",
        })

    def _config(self, name, family="titans", world_size=2, resume=None, shared_cache=True):
        model = {
            "type": family,
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2 if family == "titans" else 1,
            "num_attention_heads": 2,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 0,
            "use_cache": False,
        }
        if family == "titans":
            model.update({
                "variant": "mac",
                "num_key_value_heads": 1,
                "memory_num_heads": 2,
                "memory_head_dim": 4,
                "memory_hidden_size": 8,
                "memory_num_layers": 2,
                "memory_chunk_size": 3,
                "mac_segment_size": 4,
                "persistent_memory_tokens": 2,
                "sliding_window": 3,
                "qkv_conv_kernel": 2,
                "attn_implementation": "eager",
                "attention_dropout": 0.0,
                "resid_dropout": 0.0,
                "use_parallel_memory_training": True,
            })
        else:
            model.update({"ttt_layer_type": "linear", "mini_batch_size": 2})
        training = {
            "output_dir": str(self.directory / name),
            "use_cpu": True,
            "ddp_timeout": 60,
            "ddp_find_unused_parameters": False,
            "activation_checkpointing": True,
            "per_device_train_batch_size": 4 // (world_size * 2),
            "gradient_accumulation_steps": 2,
            "average_tokens_across_devices": True,
            "train_sampling_strategy": "sequential",
            "max_steps": 2,
            "optim": "sgd",
            "optim_args": "momentum=0.9",
            "learning_rate": 0.05,
            "lr_scheduler_type": "constant",
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "max_grad_norm": 0.0,
            "logging_strategy": "steps",
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 2,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "disable_tqdm": True,
            "report_to": [],
            "seed": _SEED,
            "data_seed": _SEED,
        }
        if resume is not None:
            training["resume_from_checkpoint"] = str(resume)
        config = {
            "model": model,
            "tokenizer": str(self.tokenizer_path),
            "dataset": {
                "name": str(self.dataset),
                "max_seq_length": 8,
                "shared_cache": shared_cache,
            },
            "training_args": training,
        }
        path = self.directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        return path, Path(training["output_dir"])

    def _launch(self, name, commands):
        processes = []
        logs = []
        try:
            for rank, (command, environment) in enumerate(commands):
                path = self.directory / f"{name}-worker-{rank}.log"
                logs.append(path)
                with path.open("wb") as stream:
                    processes.append(subprocess.Popen(
                        command, cwd=_ROOT, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    ))
            deadline = time.monotonic() + _TIMEOUT
            while True:
                statuses = [process.poll() for process in processes]
                if any(status not in (None, 0) for status in statuses):
                    self.fail(f"{name} workers failed: {statuses}")
                if all(status == 0 for status in statuses):
                    break
                if time.monotonic() >= deadline:
                    self.fail(f"{name} exceeded {_TIMEOUT}s")
                time.sleep(0.1)
        except BaseException as error:
            _terminate_groups(processes)
            diagnostics = "\n".join(
                f"--- {path.name} ---\n{path.read_text(errors='replace')}" for path in logs
            )
            raise AssertionError(f"{error}\n{diagnostics}") from error

    def _torchrun(self, name, config):
        environment = self.environment.copy()
        # Real torchrun ranks must override stale metadata inherited from SLURM.
        environment.update({
            "SLURM_PROCID": "17", "SLURM_LOCALID": "9", "SLURM_NTASKS": "40",
            "SLURM_NODEID": "8", "SLURM_NNODES": "10", "SLURM_JOB_ID": "70123",
        })
        self._launch(name, [([
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc-per-node=2", "--max-restarts=0",
            str(_ROOT / "train.py"), str(config),
        ], environment)])

    def _slurm(self, name, config):
        port = str(_free_port())
        commands = []
        for rank in range(2):
            environment = self.environment.copy()
            environment.update({
                "SLURM_PROCID": str(rank),
                "SLURM_LOCALID": str(rank),
                "SLURM_NTASKS": "2",
                "SLURM_NTASKS_PER_NODE": "2",
                "SLURM_NODEID": "0",
                "SLURM_NNODES": "1",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_JOB_ID": "70124",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": port,
            })
            commands.append(([sys.executable, str(_ROOT / "train.py"), str(config)], environment))
        self._launch(name, commands)

    def _checkpoint(self, output, step, world_size, family):
        path = output / f"checkpoint-{step}"
        self.assertTrue(path.is_dir(), path)
        state = json.loads((path / "trainer_state.json").read_text())
        self.assertEqual(state["global_step"], step)
        self.assertEqual(state["max_steps"], 2)
        self.assertEqual(state["is_world_process_zero"], True)
        losses = [entry["loss"] for entry in state["log_history"] if "loss" in entry]
        self.assertEqual(len(losses), step)
        self.assertTrue(all(math.isfinite(loss) and loss > 0 for loss in losses))
        self.assertEqual(json.loads((path / "config.json").read_text())["model_type"], family)
        expected_rng = {"rng_state.pth"} if world_size == 1 else {"rng_state_0.pth", "rng_state_1.pth"}
        self.assertEqual({item.name for item in path.glob("rng_state*.pth")}, expected_rng)
        arguments = torch.load(path / "training_args.bin", map_location="cpu", weights_only=False)
        self.assertTrue(arguments.use_cpu)
        self.assertTrue(arguments.activation_checkpointing)
        self.assertFalse(arguments.ddp_find_unused_parameters)
        self.assertEqual(arguments.max_steps, 2)
        optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        self.assertTrue(optimizer["state"], "SGD momentum must be checkpointed")
        self.assertTrue(all("momentum_buffer" in value for value in optimizer["state"].values()))
        weights = load_file(path / "model.safetensors", device="cpu")
        self.assertTrue(weights)
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in weights.values()))
        return path, state, optimizer, weights

    def _assert_nested_equal(self, observed, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(observed.keys(), expected.keys())
            for key in expected:
                self._assert_nested_equal(observed[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(observed), len(expected))
            for actual, wanted in zip(observed, expected):
                self._assert_nested_equal(actual, wanted)
        else:
            self.assertEqual(observed, expected)

    def _cache_snapshot(self):
        caches = sorted(self.cache.rglob("cache-*.arrow"))
        self.assertEqual(len(caches), 1, f"Expected one shared tokenization cache, found {caches}")
        processed = Dataset.from_file(str(caches[0]))
        self.assertEqual(len(processed), len(_LENGTHS))
        for row, ids in zip(processed, self.expected_ids):
            padding = 8 - len(ids)
            self.assertEqual(row["input_ids"], ids + [0] * padding)
            self.assertEqual(row["attention_mask"], [1] * len(ids) + [0] * padding)
            self.assertEqual(row["labels"], ids + [-100] * padding)
        return {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in caches}

    def test_torchrun_updates_match_global_batch_and_checkpoint_resume(self):
        distributed_config, distributed_output = self._config("distributed")
        self._torchrun("distributed", distributed_config)
        first, _, _, first_weights = self._checkpoint(distributed_output, 1, 2, "titans")
        final, final_state, final_optimizer, final_weights = self._checkpoint(distributed_output, 2, 2, "titans")
        self.assertTrue(any(not torch.equal(first_weights[key], final_weights[key]) for key in final_weights))
        cache = self._cache_snapshot()

        single_config, single_output = self._config("single", world_size=1)
        self._launch("single", [([
            sys.executable, "-m", "accelerate.commands.launch",
            "--cpu", "--num_processes=1", "--num_machines=1",
            "--mixed_precision=no", "--dynamo_backend=no",
            str(_ROOT / "train.py"), str(single_config),
        ], self.environment)])
        for step in (1, 2):
            _, _, _, expected = self._checkpoint(single_output, step, 1, "titans")
            observed = first_weights if step == 1 else final_weights
            self.assertEqual(observed.keys(), expected.keys())
            for key in expected:
                torch.testing.assert_close(observed[key], expected[key], rtol=1e-4, atol=2e-6, msg=key)
        self.assertEqual(self._cache_snapshot(), cache, "Baseline must reuse, not rewrite, preprocessing")

        resumed_config, resumed_output = self._config("resumed", resume=first)
        self._torchrun("resumed", resumed_config)
        resumed, resumed_state, resumed_optimizer, resumed_weights = self._checkpoint(resumed_output, 2, 2, "titans")
        self._assert_nested_equal(resumed_weights, final_weights)
        self._assert_nested_equal(resumed_optimizer, final_optimizer)
        for key in ("global_step", "max_steps", "epoch", "num_train_epochs"):
            self.assertEqual(resumed_state[key], final_state[key], key)
        for filename in ("scheduler.pt",):
            self._assert_nested_equal(
                torch.load(resumed / filename, map_location="cpu", weights_only=True),
                torch.load(final / filename, map_location="cpu", weights_only=True),
            )
        self.assertEqual(self._cache_snapshot(), cache, "Resume must reuse preprocessing")
        self.assertFalse((resumed_output / "checkpoint-1").exists())

    def test_slurm_workers_train_ttt_with_per_node_preprocessing(self):
        config, output = self._config("slurm-ttt", family="ttt", shared_cache=False)
        self._slurm("slurm-ttt", config)
        _, _, _, first = self._checkpoint(output, 1, 2, "ttt")
        _, _, _, final = self._checkpoint(output, 2, 2, "ttt")
        self.assertEqual(first.keys(), final.keys())
        self.assertTrue(any(not torch.equal(first[key], final[key]) for key in first))
        self._cache_snapshot()


if __name__ == "__main__":
    unittest.main()
