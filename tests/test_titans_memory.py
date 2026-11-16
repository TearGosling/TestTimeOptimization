"""CPU regressions for chunk-start memory learning and its outer gradients."""

import unittest

import torch
from torch.nn import functional as F

from tto.src.models.titans.configuration_titans import TitansConfig
from tto.src.models.titans.modeling_titans import TitansNeuralMemory


STATE_KEYS = ("weights", "biases", "surprise_weights", "surprise_biases")


def memory_read(inputs, weights, biases, activation, residual):
    """Independent memory MLP, including the residual in the loss prediction."""
    functions = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh}
    hidden = inputs
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        hidden = torch.matmul(hidden, weight) + bias
        if layer + 1 < len(weights):
            hidden = functions[activation](hidden)
    return hidden + inputs if residual else hidden


def chunk_oracle(queries, keys, values, theta, eta, alpha, state, *, activation,
                 residual, loss_scale, update, output_after_update):
    """Differentiate each token's loss at W_start, then recur explicitly."""
    if not update:
        return memory_read(queries, state["weights"], state["biases"],
                           activation, residual), state

    parameters = tuple(state["weights"] + state["biases"])
    layers = len(state["weights"])
    current = state
    outputs = []
    for token in range(queries.shape[2]):
        prediction = memory_read(keys[:, :, token:token + 1], state["weights"],
                                 state["biases"], activation, residual)
        loss = 0.5 * loss_scale * (prediction - values[:, :, token:token + 1]).square().sum()
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        rate = theta[:, :, token, None, None]
        momentum = eta[:, :, token, None, None]
        retention = 1.0 - alpha[:, :, token, None, None]
        following = {key: [] for key in STATE_KEYS}
        for layer in range(layers):
            for name, gradient in (("weights", gradients[layer]),
                                   ("biases", gradients[layers + layer])):
                surprise_name = "surprise_" + name
                surprise = momentum * current[surprise_name][layer] - rate * gradient
                following[surprise_name].append(surprise)
                following[name].append(retention * current[name][layer] + surprise)
        read_state = following if output_after_update else state
        outputs.append(memory_read(queries[:, :, token:token + 1],
                                   read_state["weights"], read_state["biases"],
                                   activation, residual))
        current = following
    return torch.cat(outputs, dim=2), current


class TitansMemoryRegressionTests(unittest.TestCase):
    def make_case(self, layers=2, activation="silu", residual=True, length=4,
                  dtype=torch.float64, gate_kind="ordinary"):
        config = TitansConfig(
            hidden_size=6, intermediate_size=9, num_hidden_layers=1,
            num_attention_heads=2, memory_num_heads=2, memory_head_dim=3,
            memory_num_layers=layers, memory_hidden_size=5,
            memory_activation=activation, memory_residual=residual,
            memory_loss_scale=0.7, memory_chunk_size=length,
            variant="lmm",
            use_memory_convolution=False, use_attention_convolution=False,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(831)
            memory = TitansNeuralMemory(config, layer_idx=0).to(dtype=dtype)
        generator = torch.Generator().manual_seed(1209 + layers)

        def normal(shape, scale):
            return scale * torch.randn(shape, generator=generator, dtype=dtype)

        shape = (2, 2, length, 3)
        inputs = [normal(shape, 0.45) for _ in range(3)]
        gate_shape = shape[:-1]
        if gate_kind == "ordinary":
            gates = [0.1 + 0.6 * torch.rand(gate_shape, generator=generator, dtype=dtype)
                     for _ in range(3)]
        elif gate_kind == "boundaries":
            # Every combination includes exact zeros and ones, with repeated resets.
            patterns = ([0., 1., 1., 0., 1., 0., 1., 1.],
                        [1., 0., 1., 0., 0., 1., 1., 0.],
                        [0., 1., 0., 1., 1., 1., 0., 0.])
            gates = []
            for pattern in patterns:
                base = torch.tensor(pattern, dtype=dtype)
                gates.append(torch.stack([base.roll(index) for index in range(4)])
                             .reshape(gate_shape))
        elif gate_kind == "underflow":
            # Products underflow even in float64, although local updates stay finite.
            gates = [torch.full(gate_shape, 0.2, dtype=dtype),
                     torch.full(gate_shape, 1e-10, dtype=dtype),
                     torch.full(gate_shape, 1.0 - 1e-10, dtype=dtype)]
        else:
            raise ValueError(gate_kind)
        state = {key: [] for key in STATE_KEYS}
        dimensions = [3] + [5] * (layers - 1) + [3]
        for input_width, output_width in zip(dimensions[:-1], dimensions[1:]):
            state["weights"].append(normal((2, 2, input_width, output_width), 0.3))
            state["biases"].append(normal((2, 2, 1, output_width), 0.25))
            state["surprise_weights"].append(normal((2, 2, input_width, output_width), 0.08))
            state["surprise_biases"].append(normal((2, 2, 1, output_width), 0.06))
        return memory, inputs + gates, state

    @staticmethod
    def clone_case(inputs, state):
        inputs = [value.detach().clone().requires_grad_() for value in inputs]
        state = {key: [value.detach().clone().requires_grad_() for value in state[key]]
                 for key in STATE_KEYS}
        leaves = inputs + [value for key in STATE_KEYS for value in state[key]]
        return inputs, state, leaves

    @staticmethod
    def result_tensors(result):
        output, state = result
        return [output] + [value for key in STATE_KEYS for value in state[key]]

    @staticmethod
    def objective(tensors):
        # Unequal signed probes exercise each token/head/parameter, not just sums.
        total = tensors[0].new_zeros(())
        for index, value in enumerate(tensors):
            probe = torch.arange(value.numel(), device=value.device, dtype=value.dtype)
            probe = torch.cos(probe * 0.37 + index).reshape_as(value)
            total = total + (probe * value + 0.07 * value.square()).sum()
        return total

    def assert_result_and_outer_gradients(self, actual, expected, actual_leaves,
                                          expected_leaves, *, dtype):
        tolerance = {"rtol": 2e-9, "atol": 2e-10} if dtype == torch.float64 else {
            "rtol": 3e-4, "atol": 3e-5}
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
            # Unused controls in update=False have the same derivative as explicit zeros.
            observed = torch.zeros_like(leaf) if observed is None else observed
            reference = torch.zeros_like(leaf) if reference is None else reference
            with self.subTest(outer_gradient=index):
                torch.testing.assert_close(observed, reference, **tolerance)

    def compare_oracle(self, *, update=True, output_after_update=True, **case_options):
        memory, inputs, state = self.make_case(**case_options)
        actual_inputs, actual_state, actual_leaves = self.clone_case(inputs, state)
        expected_inputs, expected_state, expected_leaves = self.clone_case(inputs, state)
        actual = memory._parallel_chunk(*actual_inputs, actual_state, update=update,
                                        output_after_update=output_after_update)
        expected = chunk_oracle(
            *expected_inputs, expected_state, activation=memory.config.memory_activation,
            residual=memory.config.memory_residual, loss_scale=memory.config.memory_loss_scale,
            update=update, output_after_update=output_after_update,
        )
        self.assert_result_and_outer_gradients(actual, expected, actual_leaves,
                                               expected_leaves, dtype=inputs[0].dtype)

    def test_chunk_start_gradients_prefix_reads_and_outer_gradients(self):
        for layers in (1, 2, 3):
            for activation in ("silu", "gelu", "relu", "tanh"):
                for residual in (False, True):
                    for after in (False, True):
                        with self.subTest(layers=layers, activation=activation,
                                          residual=residual, output_after_update=after):
                            self.compare_oracle(layers=layers, activation=activation,
                                                residual=residual, output_after_update=after)

    def test_no_update_preserves_state_and_ignores_learning_inputs(self):
        for after in (False, True):
            for residual in (False, True):
                with self.subTest(output_after_update=after, residual=residual):
                    self.compare_oracle(layers=3, activation="gelu", residual=residual,
                                        update=False, output_after_update=after)

    def test_exact_gate_resets_preserve_values_and_boundary_derivatives(self):
        for layers in (1, 3):
            for after in (False, True):
                with self.subTest(layers=layers, output_after_update=after):
                    self.compare_oracle(layers=layers, activation="tanh", length=8,
                                        gate_kind="boundaries", output_after_update=after)

    def test_sustained_decay_remains_correct_after_product_underflow(self):
        for dtype in (torch.float64, torch.float32):
            for after in (False, True):
                with self.subTest(dtype=dtype, output_after_update=after):
                    self.compare_oracle(layers=2, activation="silu", length=40,
                                        dtype=dtype, gate_kind="underflow",
                                        output_after_update=after)

    def test_single_token_parallel_matches_online_values_and_outer_gradients(self):
        for layers in (1, 2, 3):
            for after in (False, True):
                with self.subTest(layers=layers, output_after_update=after):
                    memory, inputs, state = self.make_case(layers=layers, length=1,
                                                           activation="gelu", residual=False)
                    parallel_inputs, parallel_state, parallel_leaves = self.clone_case(inputs, state)
                    online_inputs, online_state, online_leaves = self.clone_case(inputs, state)
                    parallel = memory._parallel_chunk(*parallel_inputs, parallel_state, update=True,
                                                      output_after_update=after)
                    online = memory._sequential_chunk(*online_inputs, online_state, update=True,
                                                      output_after_update=after)
                    self.assert_result_and_outer_gradients(parallel, online, parallel_leaves,
                                                           online_leaves, dtype=inputs[0].dtype)


if __name__ == "__main__":
    unittest.main()
