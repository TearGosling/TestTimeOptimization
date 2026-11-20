"""Two-rank CPU training must match a single-process global batch."""

import copy
from datetime import timedelta
from pathlib import Path
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tto import TitansConfig, TitansForCausalLM


_WORLD_SIZE = 2
_JOIN_TIMEOUT = 180


def _make_model(variant):
    # Only child processes change thread settings or seed the CPU generator.
    torch.random.default_generator.manual_seed(2501)
    config = TitansConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        memory_num_heads=2, memory_head_dim=8, memory_hidden_size=12,
        memory_num_layers=2, memory_chunk_size=3, mac_segment_size=4,
        persistent_memory_tokens=2, sliding_window=3, qkv_conv_kernel=2,
        variant=variant, attn_implementation="eager", use_cache=False,
        attention_dropout=0.0, resid_dropout=0.0,
        use_parallel_memory_training=True,
    )
    return TitansForCausalLM(config).cpu().train()


def _global_batch(step):
    # Ranks see disjoint tokens. Seven tokens span complete and partial segments.
    ids = (torch.arange(14).reshape(_WORLD_SIZE, 7) + 3 * step) % 28 + 4
    labels = ids.clone()
    if step == 0:
        labels[0, [2, 5, 6]] = -100
        labels[1, 3] = -100
    else:
        labels[0, [1, 3, 4, 6]] = -100
        labels[1, 2] = -100
    return ids, labels


def _assert_identical_across_ranks(tensor, description):
    gathered = [torch.empty_like(tensor) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, tensor)
    for rank, other in enumerate(gathered):
        torch.testing.assert_close(
            tensor, other, rtol=0, atol=0,
            msg=f"{description}: rank {rank} differs after synchronization",
        )


def _critical_parameters(model):
    # These pre-norms used to be unused in MAG/MAC. Check every fusion weight too.
    names = set()
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}."
        names.add(prefix + "memory_norm.weight")
        names.add(prefix + "attn_norm.weight")
        names.update(prefix + "fusion." + name for name, _ in layer.fusion.named_parameters())
    return names


def _run_mode(rank, variant, checkpointing):
    case = unittest.TestCase()
    description = f"{variant}, checkpointing={checkpointing}"
    model = _make_model(variant)
    # The reference deliberately stays uncheckpointed, including in checkpointed runs.
    reference = copy.deepcopy(model)
    if checkpointing:
        model.activation_checkpointing_enable(
            activation_checkpointing_kwargs={"use_reentrant": False},
        )
    for layer in model.model.layers:
        case.assertEqual(layer.gradient_checkpointing, checkpointing, description)
    ddp = DistributedDataParallel(model, find_unused_parameters=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    critical = _critical_parameters(model)
    parameters = dict(model.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    case.assertEqual(parameters.keys(), reference_parameters.keys(), description)
    case.assertTrue(critical.issubset(parameters), description)

    # A second forward catches unfinished reductions from the preceding backward.
    for step in range(2):
        context = f"{description}, step={step}"
        ids, labels = _global_batch(step)
        local_ids = ids[rank:rank + 1]
        local_labels = labels[rank:rank + 1]
        case.assertFalse(torch.equal(ids[0], ids[1]), context)
        token_counts = labels[:, 1:].ne(-100).sum(dim=1)
        case.assertNotEqual(token_counts[0].item(), token_counts[1].item(), context)
        num_items = local_labels[:, 1:].ne(-100).sum()
        dist.all_reduce(num_items, op=dist.ReduceOp.SUM)
        case.assertEqual(num_items.item(), token_counts.sum().item(), context)

        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        if step == 0:
            # Demonstrate that these shards really produce different gradients.
            # Without all-reduce, neither local result can stand in for the global one.
            local_probe = copy.deepcopy(reference)
            probe_loss = local_probe(
                local_ids, labels=local_labels, use_cache=False,
                num_items_in_batch=num_items,
            ).loss * _WORLD_SIZE
            local_gradient, = torch.autograd.grad(
                probe_loss, local_probe.model.embed_tokens.weight,
            )
            shard_gradients = [torch.empty_like(local_gradient) for _ in range(_WORLD_SIZE)]
            dist.all_gather(shard_gradients, local_gradient.detach())
            case.assertFalse(
                torch.allclose(shard_gradients[0], shard_gradients[1], rtol=2e-4, atol=2e-6),
                f"{context}: rank-local gradients must differ before synchronization",
            )
            del local_probe, probe_loss

        expected_loss = reference(
            ids, labels=labels, use_cache=False, num_items_in_batch=num_items,
        ).loss
        expected_loss.backward()
        local_loss = ddp(
            local_ids, labels=local_labels, use_cache=False,
            num_items_in_batch=num_items,
        ).loss
        # Each shard contributes its loss sum / GLOBAL valid-target count.
        # DDP averages gradients, so cancel that average explicitly.
        (local_loss * _WORLD_SIZE).backward()
        loss_sum = local_loss.detach().clone()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(loss_sum, expected_loss.detach(), rtol=2e-5, atol=2e-6)

        gradients = []
        for name, parameter in parameters.items():
            expected_gradient = reference_parameters[name].grad
            case.assertIsNotNone(expected_gradient, f"{context}: reference {name}")
            case.assertIsNotNone(parameter.grad, f"{context}: distributed {name}")
            case.assertTrue(torch.isfinite(parameter.grad).all().item(), f"{context}: {name}")
            torch.testing.assert_close(
                parameter.grad, expected_gradient, rtol=2e-4, atol=2e-6,
                msg=f"{context}: gradient for {name} does not match global batch",
            )
            if name in critical:
                case.assertGreater(
                    parameter.grad.abs().max().item(), 0,
                    f"{context}: {name} must receive a nonzero outer gradient",
                )
            gradients.append(parameter.grad.detach().reshape(-1))
        _assert_identical_across_ranks(torch.cat(gradients), f"{context}: gradients")
        if step == 0:
            global_gradient = parameters["model.embed_tokens.weight"].grad
            for shard_gradient in shard_gradients:
                case.assertFalse(
                    torch.allclose(global_gradient, shard_gradient, rtol=2e-4, atol=2e-6),
                    f"{context}: synchronized gradient must differ from each local shard",
                )

        before_step = model.model.embed_tokens.weight.detach().clone()
        optimizer.step()
        reference_optimizer.step()
        case.assertFalse(torch.equal(before_step, model.model.embed_tokens.weight), context)
        for name, parameter in parameters.items():
            torch.testing.assert_close(
                parameter, reference_parameters[name], rtol=2e-5, atol=2e-6,
                msg=f"{context}: optimizer-updated {name} does not match global batch",
            )
        _assert_identical_across_ranks(
            torch.cat([parameter.detach().reshape(-1) for parameter in parameters.values()]),
            f"{context}: optimizer-updated weights",
        )
    dist.barrier()


def _ddp_worker(rank, init_method):
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            backend="gloo", init_method=init_method,
            rank=rank, world_size=_WORLD_SIZE, timeout=timedelta(seconds=45),
        )
        # Reuse one process group for the complete mode matrix.
        for variant in ("mag", "mac"):
            for checkpointing in (False, True):
                _run_mode(rank, variant, checkpointing)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _stop_workers(context):
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 5
    for process in context.processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in context.processes:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + 5
    for process in context.processes:
        process.join(timeout=max(0, deadline - time.monotonic()))


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(),
    "torch.distributed with Gloo is required",
)
class TitansDDPTests(unittest.TestCase):
    def test_two_rank_training_matches_global_batch_with_and_without_checkpointing(self):
        # Spawn isolates RNG/thread state. No environment variables or CUDA state change.
        with tempfile.TemporaryDirectory(prefix="titans-ddp-") as directory:
            init_method = (Path(directory) / "rendezvous").resolve().as_uri()
            context = mp.spawn(
                _ddp_worker, args=(init_method,), nprocs=_WORLD_SIZE, join=False,
            )
            deadline = time.monotonic() + _JOIN_TIMEOUT
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail(f"Titans DDP workers exceeded {_JOIN_TIMEOUT} seconds")
                    if context.join(timeout=min(1, remaining)):
                        break
            finally:
                _stop_workers(context)


if __name__ == "__main__":
    unittest.main()
