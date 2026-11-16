"""Multi-chunk neural-memory outputs, boundary states and outer derivatives."""

import copy
import unittest

import torch
from torch.nn import functional as F

from tto.src.models.titans.configuration_titans import TitansConfig
from tto.src.models.titans.modeling_titans import TitansNeuralMemory


STATE_KEYS = ("weights", "biases", "surprise_weights", "surprise_biases")


def read_mlp(inputs, state, residual):
    hidden = inputs
    for index, (weight, bias) in enumerate(zip(state["weights"], state["biases"])):
        hidden = hidden @ weight + bias
        if index + 1 < len(state["weights"]):
            hidden = F.silu(hidden)
    return hidden + inputs if residual else hidden


def sequence_oracle(memory, hidden_states, state, output_after_update):
    """Use autograd at each chunk boundary, with explicit token recurrences."""
    queries, keys, values, theta, eta, alpha = memory._project(hidden_states, None, False)
    outputs = []
    current = state
    for start in range(0, hidden_states.shape[1], memory.chunk_size):
        boundary = current
        parameters = tuple(boundary["weights"] + boundary["biases"])
        layers = len(boundary["weights"])
        for token in range(start, min(start + memory.chunk_size, hidden_states.shape[1])):
            prediction = read_mlp(keys[:, :, token:token + 1], boundary,
                                  memory.config.memory_residual)
            loss = 0.5 * memory.config.memory_loss_scale * (
                prediction - values[:, :, token:token + 1]
            ).square().sum()
            gradients = torch.autograd.grad(loss, parameters, create_graph=True)
            following = {name: [] for name in STATE_KEYS}
            for layer in range(layers):
                for name, gradient in (("weights", gradients[layer]),
                                       ("biases", gradients[layers + layer])):
                    surprise_name = "surprise_" + name
                    surprise = (eta[:, :, token, None, None] * current[surprise_name][layer]
                                - theta[:, :, token, None, None] * gradient)
                    following[surprise_name].append(surprise)
                    following[name].append(
                        (1.0 - alpha[:, :, token, None, None]) * current[name][layer] + surprise
                    )
            current = following
            read_state = current if output_after_update else boundary
            outputs.append(read_mlp(queries[:, :, token:token + 1], read_state,
                                    memory.config.memory_residual))
    output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(
        hidden_states.shape[0], hidden_states.shape[1], memory.memory_width
    )
    output = memory.out_norm(output) * F.silu(memory.gate_proj(hidden_states))
    return memory.out_proj(output), current


class TitansChunkBoundaryTests(unittest.TestCase):
    @staticmethod
    def result_tensors(result):
        output, state = result
        return [output] + [value for name in STATE_KEYS for value in state[name]]

    @staticmethod
    def objective(tensors):
        total = tensors[0].new_zeros(())
        for index, tensor in enumerate(tensors):
            probe = torch.arange(tensor.numel(), dtype=tensor.dtype).reshape_as(tensor)
            probe = torch.cos(probe * 0.41 + index)
            total = total + (probe * tensor + 0.03 * tensor.square()).sum()
        return total

    def compare_sequence(self, *, length, chunk_size, layers, dtype, after, residual=True):
        config = TitansConfig(
            hidden_size=4, intermediate_size=8, num_hidden_layers=1,
            num_attention_heads=2, memory_num_heads=2, memory_head_dim=2,
            memory_hidden_size=3, memory_num_layers=layers,
            memory_chunk_size=chunk_size, memory_loss_scale=0.7,
            memory_residual=residual, use_memory_convolution=False,
            use_attention_convolution=False, use_parallel_memory_training=True,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1503)
            actual_memory = TitansNeuralMemory(config, 0).to(dtype=dtype)
            with torch.no_grad():
                for weight in actual_memory.weights:
                    weight.normal_(std=0.25)
                for bias in actual_memory.biases:
                    bias.normal_(std=0.1)
                actual_memory.control_proj.weight.normal_(std=0.1)
                actual_memory.control_proj.bias.copy_(
                    torch.tensor([-3.0, 2.0, -2.0], dtype=dtype).repeat(2)
                )
            hidden = 0.3 * torch.randn(1, length, 4, dtype=dtype)
            boundary = actual_memory.initial_state(1, dtype=dtype)
            momentum = {
                name: [0.04 * torch.randn_like(value) for value in boundary[name]]
                for name in ("surprise_weights", "surprise_biases")
            }
        expected_memory = copy.deepcopy(actual_memory)

        def prepare(memory):
            inputs = hidden.clone().requires_grad_()
            state = memory.initial_state(1, dtype=dtype)
            for name, tensors in momentum.items():
                state[name] = [tensor.clone().requires_grad_() for tensor in tensors]
            leaves = [inputs] + list(memory.parameters()) + [
                tensor for name in momentum for tensor in state[name]
            ]
            return inputs, state, leaves

        actual_inputs, actual_state, actual_leaves = prepare(actual_memory)
        expected_inputs, expected_state, expected_leaves = prepare(expected_memory)
        actual = actual_memory(actual_inputs, state=actual_state, output_after_update=after)
        expected = sequence_oracle(expected_memory, expected_inputs, expected_state, after)
        # RMSNorm intentionally accumulates in fp32, including for float64 inputs.
        tolerance = {"rtol": 5e-6, "atol": 5e-7} if dtype == torch.float64 else {
            "rtol": 5e-4, "atol": 5e-5
        }
        actual_tensors = self.result_tensors(actual)
        expected_tensors = self.result_tensors(expected)
        for index, (observed, reference) in enumerate(zip(actual_tensors, expected_tensors)):
            with self.subTest(result_tensor=index):
                torch.testing.assert_close(observed, reference, **tolerance)
        actual_gradients = torch.autograd.grad(self.objective(actual_tensors), actual_leaves,
                                               allow_unused=True)
        expected_gradients = torch.autograd.grad(self.objective(expected_tensors), expected_leaves,
                                                 allow_unused=True)
        for index, (observed, reference, leaf) in enumerate(zip(
                actual_gradients, expected_gradients, actual_leaves)):
            observed = torch.zeros_like(leaf) if observed is None else observed
            reference = torch.zeros_like(leaf) if reference is None else reference
            with self.subTest(outer_gradient=index):
                torch.testing.assert_close(observed, reference, **tolerance)

    def test_nondivisible_sequences_match_chunk_boundary_oracle(self):
        for length, chunk_size, layers in ((7, 3, 2), (11, 4, 3)):
            for after in (False, True):
                for residual in (False, True):
                    with self.subTest(length=length, chunk_size=chunk_size,
                                      layers=layers, after=after, residual=residual):
                        self.compare_sequence(length=length, chunk_size=chunk_size,
                                              layers=layers, dtype=torch.float64,
                                              after=after, residual=residual)

    def test_longer_chunks_with_float32_controls_match_oracle(self):
        for length, chunk_size in ((35, 16), (67, 64)):
            with self.subTest(length=length, chunk_size=chunk_size):
                self.compare_sequence(length=length, chunk_size=chunk_size,
                                      layers=2, dtype=torch.float32, after=True)


if __name__ == "__main__":
    unittest.main()
