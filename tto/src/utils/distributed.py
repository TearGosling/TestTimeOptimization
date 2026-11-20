"""Translate direct SLURM workers into the environment consumed by Trainer."""

import ipaddress
import os
import re


_NATIVE_RANKS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


def _integer(name, value, *, minimum=0):
    if not re.fullmatch(r"[0-9]+", value) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _first(names):
    for name in names:
        if name in os.environ:
            return name, os.environ[name]
    return None, None


def _local_topology(world_size):
    node_name, node_value = _first(
        ("SLURM_STEP_NUM_NODES", "SLURM_NNODES", "SLURM_JOB_NUM_NODES")
    )
    nodes = _integer(node_name, node_value, minimum=1) if node_name else None
    node_id = (
        _integer("SLURM_NODEID", os.environ["SLURM_NODEID"])
        if "SLURM_NODEID" in os.environ else None
    )
    name, value = _first(
        ("SLURM_STEP_TASKS_PER_NODE", "SLURM_TASKS_PER_NODE", "SLURM_NTASKS_PER_NODE")
    )
    if name is None:
        if nodes == 1 or world_size == 1:
            local_size = world_size
            nodes = 1
        else:
            raise ValueError(
                "Direct SLURM workers need SLURM_TASKS_PER_NODE (or "
                "SLURM_NTASKS_PER_NODE), or an explicit single-node count"
            )
    elif name == "SLURM_NTASKS_PER_NODE":
        local_size = _integer(name, value, minimum=1)
        if world_size % local_size:
            raise ValueError(f"{name} does not divide the SLURM worker count")
        inferred_nodes = world_size // local_size
        if nodes is not None and nodes != inferred_nodes:
            raise ValueError(f"{name} conflicts with the SLURM node count")
        nodes = inferred_nodes
    else:
        # Keep compact runs compressed, including on unusually large allocations.
        runs = []
        for item in value.split(","):
            match = re.fullmatch(r"([0-9]+)(?:\(x([0-9]+)\))?", item.strip())
            if match is None:
                raise ValueError(f"Malformed {name}: {value!r}")
            count = _integer(name, match[1], minimum=1)
            repeat = _integer(name, match[2] or "1", minimum=1)
            runs.append((count, repeat))
        listed_nodes = sum(repeat for _, repeat in runs)
        if sum(count * repeat for count, repeat in runs) != world_size:
            raise ValueError(f"{name} does not sum to the SLURM worker count")
        if nodes is not None and nodes != listed_nodes:
            raise ValueError(f"{name} conflicts with the SLURM node count")
        nodes = listed_nodes
        if node_id is None and len({count for count, _ in runs}) > 1:
            raise ValueError(f"SLURM_NODEID is required for heterogeneous {name}")
        remaining = node_id or 0
        local_size = runs[0][0]
        for count, repeat in runs:
            if remaining < repeat:
                local_size = count
                break
            remaining -= repeat
    if node_id is not None and node_id >= nodes:
        raise ValueError("SLURM_NODEID is outside the SLURM node count")
    return local_size, nodes


def _is_loopback(address):
    if address.lower().rstrip(".") in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(address.strip("[]")).is_loopback
    except ValueError:
        return False


def configure_distributed_environment(*, use_cpu: bool = False) -> None:
    """Configure existing direct ``srun`` workers, without spawning processes.

    A complete native RANK/WORLD_SIZE/LOCAL_RANK environment takes precedence
    unchanged. An allocation without SLURM_PROCID/SLURM_LOCALID is also untouched.
    Actual SLURM workers require both rank markers, a task count, and sufficient
    per-node topology. Multi-node workers require a common non-loopback MASTER_ADDR.
    Validation finishes before any environment changes are made.

    Pass TrainingArguments' ``use_cpu`` setting to allow CPU workers even when
    CUDA_VISIBLE_DEVICES restricts the visible GPU set. For GPU workers Accelerate
    1.12 uses the local rank as DDP's device ordinal, so all local ranks must be
    addressable. Restricting each task to one GPU is unsupported for that layout.
    """
    if all(name in os.environ for name in _NATIVE_RANKS):
        return
    if not any(name in os.environ for name in ("SLURM_PROCID", "SLURM_LOCALID")):
        return
    if any(name in os.environ for name in _NATIVE_RANKS):
        raise ValueError(
            "Incomplete native RANK/WORLD_SIZE/LOCAL_RANK environment for a SLURM worker"
        )
    missing = [name for name in ("SLURM_PROCID", "SLURM_LOCALID") if name not in os.environ]
    if missing:
        raise ValueError(f"Incomplete SLURM worker environment: missing {', '.join(missing)}")
    count_name, count = _first(("SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"))
    if count_name is None:
        raise ValueError("SLURM workers require SLURM_STEP_NUM_TASKS or SLURM_NTASKS")
    rank = _integer("SLURM_PROCID", os.environ["SLURM_PROCID"])
    local_rank = _integer("SLURM_LOCALID", os.environ["SLURM_LOCALID"])
    world_size = _integer(count_name, count, minimum=1)
    local_size, nodes = _local_topology(world_size)
    if rank >= world_size:
        raise ValueError("SLURM_PROCID must be smaller than the SLURM worker count")
    if local_rank >= local_size:
        raise ValueError("SLURM_LOCALID must be smaller than the per-node task count")
    if "LOCAL_WORLD_SIZE" in os.environ:
        existing_size = _integer("LOCAL_WORLD_SIZE", os.environ["LOCAL_WORLD_SIZE"], minimum=1)
        if existing_size != local_size:
            raise ValueError("LOCAL_WORLD_SIZE conflicts with the SLURM per-node task count")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not use_cpu and visible is not None and visible.strip() not in ("", "-1"):
        devices = [device.strip() for device in visible.split(",")]
        if any(not device for device in devices) or len(devices) < local_size:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must expose at least one GPU per local SLURM task. "
                "Accelerate 1.12 DDP uses LOCAL_RANK as its device ordinal. Expose all "
                "node GPUs to each task, use a native torchrun/accelerate launcher, "
                "or select use_cpu: true for CPU training. LOCAL_RANK cannot be reset to 0."
            )

    master = os.environ.get("MASTER_ADDR")
    if master is not None and not master.strip():
        raise ValueError("MASTER_ADDR must not be empty")
    if nodes > 1 and (master is None or _is_loopback(master)):
        raise ValueError("Multi-node SLURM workers require an explicit common, non-loopback MASTER_ADDR")
    port = os.environ.get("MASTER_PORT", "29500")
    if _integer("MASTER_PORT", port, minimum=1) > 65535:
        raise ValueError("MASTER_PORT must be <= 65535")
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(local_rank),
        LOCAL_WORLD_SIZE=str(local_size), MASTER_ADDR=master or "127.0.0.1", MASTER_PORT=port,
    )
