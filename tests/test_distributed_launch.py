"""Launcher environment contracts for Trainer-based distributed training."""

import importlib
import os
import unittest
from unittest.mock import patch

from tto.src.utils import distributed


class DistributedEnvironmentTests(unittest.TestCase):
    def configure(self, environment, *, use_cpu=False):
        with patch.dict(os.environ, environment, clear=True):
            result = distributed.configure_distributed_environment(use_cpu=use_cpu)
            self.assertIsNone(result)
            return dict(os.environ)

    def worker(self, **overrides):
        environment = {
            "SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "2",
            "SLURM_NNODES": "1", "SLURM_TASKS_PER_NODE": "2",
        }
        environment.update(overrides)
        return environment

    def assert_invalid(self, environment, message):
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, message):
                distributed.configure_distributed_environment()
            self.assertEqual(dict(os.environ), environment, "Validation must not partially change the environment")

    def test_import_has_no_environment_side_effects(self):
        environment = self.worker()
        with patch.dict(os.environ, environment, clear=True):
            importlib.reload(distributed)
            self.assertEqual(dict(os.environ), environment)

    def test_plain_script_and_allocation_are_unchanged(self):
        for environment in (
            {},
            {"CUDA_VISIBLE_DEVICES": "0", "MASTER_PORT": "30291"},
            {"SLURM_JOB_ID": "42", "SLURM_NNODES": "2", "SLURM_NTASKS": "8",
             "SLURM_TASKS_PER_NODE": "4(x2)"},
            {"LOCAL_RANK": "0"},
        ):
            with self.subTest(environment=environment):
                self.assertEqual(self.configure(environment), environment)

    def test_complete_native_launcher_wins_over_conflicting_slurm(self):
        environment = self.worker(
            RANK="3", WORLD_SIZE="8", LOCAL_RANK="1", LOCAL_WORLD_SIZE="2",
            SLURM_PROCID="not-an-integer", SLURM_NTASKS="99", SLURM_NODEID="99",
            CUDA_VISIBLE_DEVICES="0", MASTER_ADDR="native-master", MASTER_PORT="30291",
        )
        self.assertEqual(self.configure(environment), environment)

    def test_single_node_ranks_and_rendezvous(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                original = self.worker(SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank))
                result = self.configure(original)
                self.assertEqual(result["RANK"], str(rank))
                self.assertEqual(result["LOCAL_RANK"], str(rank))
                self.assertEqual(result["WORLD_SIZE"], "2")
                self.assertEqual(result["LOCAL_WORLD_SIZE"], "2")
                self.assertIn(result["MASTER_ADDR"], ("127.0.0.1", "localhost", "::1"))
                self.assertTrue(1 <= int(result["MASTER_PORT"]) <= 65535)
                self.assertTrue(all(result[key] == value for key, value in original.items()))

    def test_single_node_can_derive_local_count_from_node_count(self):
        environment = self.worker()
        del environment["SLURM_TASKS_PER_NODE"]
        self.assertEqual(self.configure(environment)["LOCAL_WORLD_SIZE"], "2")
        self.assertEqual(self.configure({
            "SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "1",
        })["LOCAL_WORLD_SIZE"], "1")

    def test_compact_heterogeneous_multinode_counts(self):
        for node, rank, local_rank, local_size in ((0, 0, 0, 4), (1, 7, 3, 4), (2, 9, 1, 2)):
            with self.subTest(node=node):
                result = self.configure(self.worker(
                    SLURM_PROCID=str(rank), SLURM_LOCALID=str(local_rank), SLURM_NTASKS="10",
                    SLURM_NNODES="3", SLURM_NODEID=str(node), SLURM_TASKS_PER_NODE="4(x2),2",
                    MASTER_ADDR="training-node-0", MASTER_PORT="30291",
                ))
                self.assertEqual(result["RANK"], str(rank))
                self.assertEqual(result["WORLD_SIZE"], "10")
                self.assertEqual(result["LOCAL_RANK"], str(local_rank))
                self.assertEqual(result["LOCAL_WORLD_SIZE"], str(local_size))
                self.assertEqual(result["MASTER_ADDR"], "training-node-0")
                self.assertEqual(result["MASTER_PORT"], "30291")

    def test_uniform_counts_do_not_require_node_id(self):
        result = self.configure(self.worker(
            SLURM_PROCID="5", SLURM_LOCALID="1", SLURM_NTASKS="8", SLURM_NNODES="2",
            SLURM_TASKS_PER_NODE="4(x2)", MASTER_ADDR="training-node-0",
        ))
        self.assertEqual(result["LOCAL_WORLD_SIZE"], "4")
        environment = self.worker(SLURM_NTASKS="8", SLURM_NNODES="2", MASTER_ADDR="training-node-0")
        del environment["SLURM_TASKS_PER_NODE"]
        environment["SLURM_NTASKS_PER_NODE"] = "4"
        self.assertEqual(self.configure(environment)["LOCAL_WORLD_SIZE"], "4")

    def test_step_metadata_precedes_allocation_metadata(self):
        result = self.configure(self.worker(
            SLURM_PROCID="1", SLURM_LOCALID="1", SLURM_NTASKS="16", SLURM_NNODES="4",
            SLURM_TASKS_PER_NODE="4(x4)", SLURM_STEP_NUM_TASKS="2",
            SLURM_STEP_NUM_NODES="1", SLURM_STEP_TASKS_PER_NODE="2",
        ))
        self.assertEqual(result["WORLD_SIZE"], "2")
        self.assertEqual(result["LOCAL_WORLD_SIZE"], "2")

    def test_rendezvous_and_gpu_mask_are_preserved(self):
        environment = self.worker(
            MASTER_ADDR="caller-selected-host", MASTER_PORT="30291", CUDA_VISIBLE_DEVICES="GPU-a,GPU-b",
        )
        result = self.configure(environment)
        for key in ("MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"):
            self.assertEqual(result[key], environment[key])

    def test_one_visible_gpu_never_collapses_local_rank(self):
        for rank in (0, 1):
            environment = self.worker(
                SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank), CUDA_VISIBLE_DEVICES="0",
            )
            self.assert_invalid(environment, "LOCAL_RANK cannot be reset to 0")
            result = self.configure(environment, use_cpu=True)
            self.assertEqual(result["LOCAL_RANK"], str(rank))
            self.assertEqual(result["LOCAL_WORLD_SIZE"], "2")
            self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "0")
        for visible in ("", "-1"):
            self.assertEqual(self.configure(self.worker(CUDA_VISIBLE_DEVICES=visible))["LOCAL_WORLD_SIZE"], "2")

    def test_malformed_and_incomplete_worker_environments_are_atomic(self):
        cases = (
            ({"SLURM_LOCALID": "0"}, "missing SLURM_PROCID"),
            ({"SLURM_PROCID": "0"}, "missing SLURM_LOCALID"),
            ({"SLURM_PROCID": "0", "SLURM_LOCALID": "0"}, "SLURM_NTASKS"),
            (self.worker(RANK="0"), "Incomplete native"),
            (self.worker(SLURM_PROCID="bad"), "SLURM_PROCID"),
            (self.worker(SLURM_PROCID="2"), "SLURM_PROCID"),
            (self.worker(SLURM_LOCALID="2"), "SLURM_LOCALID"),
            (self.worker(SLURM_NTASKS="0"), "SLURM_NTASKS"),
            (self.worker(SLURM_TASKS_PER_NODE="2(x0)"), "SLURM_TASKS_PER_NODE"),
            (self.worker(SLURM_TASKS_PER_NODE="2(xa)"), "Malformed"),
            (self.worker(SLURM_TASKS_PER_NODE="1,"), "Malformed"),
            (self.worker(SLURM_TASKS_PER_NODE="1"), "does not sum"),
            (self.worker(SLURM_NNODES="2"), "conflicts"),
            (self.worker(SLURM_NODEID="1"), "SLURM_NODEID"),
            (self.worker(LOCAL_WORLD_SIZE="3"), "LOCAL_WORLD_SIZE conflicts"),
            (self.worker(MASTER_ADDR=""), "MASTER_ADDR"),
            (self.worker(MASTER_PORT=""), "MASTER_PORT"),
            (self.worker(MASTER_PORT="65536"), "MASTER_PORT"),
        )
        for environment, message in cases:
            with self.subTest(environment=environment):
                self.assert_invalid(environment, message)
        environment = self.worker()
        del environment["SLURM_TASKS_PER_NODE"]
        del environment["SLURM_NNODES"]
        self.assert_invalid(environment, "per-node|SLURM_TASKS_PER_NODE")
        self.assert_invalid(self.worker(
            SLURM_NTASKS="3", SLURM_NNODES="2", SLURM_TASKS_PER_NODE="2,1",
            MASTER_ADDR="training-node-0",
        ), "SLURM_NODEID is required")

    def test_multinode_requires_nonloopback_common_master(self):
        for master in (None, "localhost", "127.0.0.1", "::1"):
            with self.subTest(master=master):
                environment = self.worker(SLURM_NTASKS="4", SLURM_NNODES="2", SLURM_TASKS_PER_NODE="2(x2)")
                if master is not None:
                    environment["MASTER_ADDR"] = master
                self.assert_invalid(environment, "explicit common, non-loopback MASTER_ADDR")




if __name__ == "__main__":
    unittest.main()
