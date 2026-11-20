"""CPU numerical oracles for the chosen MAG/MAC pre-norm decomposition.

The paper permits nonlinear fusion and leaves pre-norm placement unspecified.
These tests pin this implementation's two-input fusion and outer residuals, not
an assertion that the paper requires a unique gate or normalization scheme.
"""

import copy
import unittest

import torch
from torch.nn import functional as F

from tto import TitansConfig, TitansForCausalLM


def rms_norm(inputs, norm):
    variance = inputs.float().square().mean(-1, keepdim=True)
    normalized = inputs.float() * torch.rsqrt(variance + norm.variance_epsilon)
    return norm.weight * normalized.to(inputs.dtype)


def fuse(layer, core, memory, skip_norm=None):
    """Evaluate the two-input gate explicitly, without calling its forward."""
    core = core if skip_norm == "fusion.core_norm" else rms_norm(core, layer.fusion.core_norm)
    memory = memory if skip_norm == "fusion.memory_norm" else rms_norm(memory, layer.fusion.memory_norm)
    gate = torch.sigmoid(F.linear(torch.cat((core, memory), -1),
                                layer.fusion.gate_proj.weight,
                                layer.fusion.gate_proj.bias) + layer.fusion.gate_bias)
    return F.linear(gate * core + (1 - gate) * memory, layer.fusion.out_proj.weight)


def finish(layer, inputs):
    normalized = rms_norm(inputs, layer.ffn_norm)
    mlp = layer.mlp
    update = F.linear(F.silu(F.linear(normalized, mlp.gate_proj.weight))
                      * F.linear(normalized, mlp.up_proj.weight), mlp.down_proj.weight)
    return inputs + update


def decompose(layer, inputs, *, skip_norm=None, output_after_update=True):
    """Compose real attention/memory branches independently of decoder forward.

    MAC uses no retrieval helper here: its frozen-state read is the ordinary
    memory forward with updates disabled. Segment state is carried explicitly.
    """
    batch, length, _ = inputs.shape
    positions = torch.arange(length, device=inputs.device)[None].expand(batch, -1)
    persistent = (None if layer.persistent_memory is None else
                  layer.persistent_memory[None].expand(batch, -1, -1))

    def normalize(value, name):
        return value if skip_norm == name else rms_norm(value, getattr(layer, name))

    if layer.variant == "mag":
        core, _ = layer.self_attn(
            normalize(inputs, "attn_norm"), position_ids=positions,
            prefix_states=None if persistent is None else normalize(persistent, "attn_norm"),
            sliding_window=layer.config.sliding_window,
        )
        memory_inputs = inputs if persistent is None else torch.cat((persistent, inputs), 1)
        memory, _ = layer.memory(normalize(memory_inputs, "memory_norm"), update_cache=False)
        memory = memory[:, 0 if persistent is None else persistent.shape[1]:]
        cores, memories = core, memory
    else:
        state = layer.memory.initial_state(batch, inputs.device, inputs.dtype)
        cores, memories = [], []
        for start in range(0, length, layer.config.mac_segment_size):
            segment = inputs[:, start:start + layer.config.mac_segment_size]
            frozen, _ = layer.memory(normalize(segment, "memory_norm"), state=state,
                                     update=False, update_cache=False)
            prefix = frozen if persistent is None else torch.cat((persistent, frozen), 1)
            prefix_mask = torch.ones(segment.shape[1], segment.shape[1],
                                     device=inputs.device, dtype=torch.bool).tril()
            prefix_mask = F.pad(prefix_mask, (0 if persistent is None else persistent.shape[1], 0),
                                value=True)[None].expand(batch, -1, -1)
            core, _ = layer.self_attn(
                normalize(segment, "attn_norm"), position_ids=positions[:, start:start + segment.shape[1]],
                prefix_states=normalize(prefix, "attn_norm"), prefix_attention_mask=prefix_mask,
                sliding_window=None, cache_attention=False,
            )
            memory, state = layer.memory(normalize(core, "memory_norm"), state=state,
                                         update=True, output_after_update=output_after_update,
                                         update_cache=False)
            cores.append(core)
            memories.append(memory)
        cores, memories = torch.cat(cores, 1), torch.cat(memories, 1)
    combined = fuse(layer, cores, memories, skip_norm)
    return finish(layer, inputs + combined), cores, memories


def objective(output):
    probe = torch.linspace(-0.7, 1.1, output.numel(), device=output.device,
                           dtype=output.dtype).reshape_as(output)
    return (probe * output + 0.13 * output.square()).sum()


class TitansVariantDecompositionTests(unittest.TestCase):
    def make_model(self, variant, *, persistent=2, convolution=False, parallel=True):
        # fork_rng keeps these deterministic tests isolated from other tests' RNG.
        with torch.random.fork_rng():
            torch.manual_seed(2571)
            config = TitansConfig(
                vocab_size=23, hidden_size=8, intermediate_size=12,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                memory_num_heads=2, memory_head_dim=4, memory_hidden_size=6,
                memory_num_layers=2, memory_chunk_size=2, mac_segment_size=3,
                persistent_memory_tokens=persistent, sliding_window=4,
                use_attention_convolution=convolution, use_memory_convolution=convolution,
                qkv_conv_kernel=2, variant=variant, attention_dropout=0.0, resid_dropout=0.0,
                use_parallel_memory_training=parallel,
            )
            config._attn_implementation = "eager"
            model = TitansForCausalLM(config)
            # Nonunit, nonuniform learned norms and visible branch updates make
            # missing norms, double normalization, and stale reads distinguishable.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "norm" in name and name.endswith("weight"):
                        parameter.copy_(torch.linspace(0.6, 1.4, parameter.numel()).reshape_as(parameter))
                    else:
                        parameter.normal_(0.0, 0.17 if parameter.ndim > 1 else 0.08)
            return model.eval()

    @staticmethod
    def inputs(length=5):
        values = torch.arange(2 * length * 8, dtype=torch.float32).reshape(2, length, 8)
        return (0.7 * torch.sin(values * 0.37) + 0.4 * torch.cos(values * 0.13) + 0.3).requires_grad_()

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=3e-6)

    def assert_distinct(self, actual, wrong):
        self.assertGreater((actual - wrong).abs().max().item(), 1e-6)

    def test_decoder_outputs_and_outer_gradients_match_explicit_branches(self):
        # Length six is two full MAC segments; length five ends in a partial one.
        for variant in ("mag", "mac"):
            for persistent in (0, 2):
                for convolution, length in ((False, 6), (True, 5)):
                    with self.subTest(variant=variant, persistent=persistent,
                                      convolution=convolution, length=length):
                        layer = self.make_model(variant, persistent=persistent,
                                                convolution=convolution).model.layers[0]
                        reference = copy.deepcopy(layer)
                        inputs = self.inputs(length)
                        reference_inputs = inputs.detach().clone().requires_grad_()
                        actual = layer(inputs, use_cache=False)[0]
                        expected, _, _ = decompose(reference, reference_inputs)
                        self.assert_close(actual, expected)
                        actual_leaves = (inputs, *layer.parameters())
                        expected_leaves = (reference_inputs, *reference.parameters())
                        actual_grads = torch.autograd.grad(objective(actual), actual_leaves)
                        expected_grads = torch.autograd.grad(objective(expected), expected_leaves)
                        for name, observed, target in zip(
                                ("inputs", *dict(layer.named_parameters())), actual_grads, expected_grads):
                            with self.subTest(parameter=name):
                                self.assert_close(observed, target)
                                self.assertTrue(torch.isfinite(observed).all().item())
                                self.assertGreater(observed.abs().max().item(), 0.0)

    def test_raw_residual_is_outside_the_two_input_gate(self):
        for variant in ("mag", "mac"):
            with self.subTest(variant=variant):
                layer = self.make_model(variant).model.layers[0]
                inputs = self.inputs()
                actual = layer(inputs, use_cache=False)[0]
                expected, core, memory = decompose(layer, inputs)
                self.assert_close(actual, expected)
                normalized_residual = finish(layer, rms_norm(inputs, layer.attn_norm)
                                             + fuse(layer, core, memory))
                residual_inside_gate = finish(layer, fuse(layer, core + inputs, memory))
                self.assert_distinct(actual, normalized_residual)
                self.assert_distinct(actual, residual_inside_gate)

    def test_each_branch_norm_changes_values_and_outer_gradients(self):
        for variant in ("mag", "mac"):
            for norm_name in ("attn_norm", "memory_norm", "fusion.core_norm", "fusion.memory_norm"):
                with self.subTest(variant=variant, norm=norm_name):
                    layer = self.make_model(variant).model.layers[0]
                    inputs = self.inputs()
                    actual = layer(inputs, use_cache=False)[0]
                    wrong, _, _ = decompose(layer, inputs, skip_norm=norm_name)
                    self.assert_distinct(actual, wrong)
                    norm = layer.get_submodule(norm_name)
                    input_grad, norm_grad = torch.autograd.grad(objective(actual), (inputs, norm.weight))
                    wrong_grad = torch.autograd.grad(objective(wrong), inputs)[0]
                    self.assertGreater(norm_grad.abs().max().item(), 1e-7)
                    self.assert_distinct(input_grad, wrong_grad)

    def test_mac_reads_after_each_causal_update_not_before_it(self):
        for parallel in (True, False):
            with self.subTest(parallel=parallel):
                layer = self.make_model("mac", parallel=parallel).model.layers[0]
                inputs = self.inputs()
                actual = layer(inputs, use_cache=False)[0]
                expected, _, memory = decompose(layer, inputs)
                stale, _, stale_memory = decompose(layer, inputs, output_after_update=False)
                self.assert_close(actual, expected)
                self.assert_distinct(memory, stale_memory)
                self.assert_distinct(actual, stale)
                # A single token isolates read/update order from segment carry.
                single = inputs[:, :1]
                updated, _, _ = decompose(layer, single)
                stale_single, _, _ = decompose(layer, single, output_after_update=False)
                self.assert_close(layer(single, use_cache=False)[0], updated)
                self.assert_distinct(updated, stale_single)

    def test_fixed_state_retrieval_matches_no_update_read_and_outer_gradients(self):
        for convolution in (False, True):
            with self.subTest(convolution=convolution):
                memory = self.make_model("mac", convolution=convolution).model.layers[0].memory
                reference = copy.deepcopy(memory)
                inputs = self.inputs()
                other_inputs = inputs.detach().clone().requires_grad_()
                # Independent leaf states test differentiation through an actual
                # learned state, rather than only the initial parameter aliases.
                state = memory.initial_state(2)
                state = {key: [(value.detach() + 0.03).requires_grad_() for value in values]
                         for key, values in state.items()}
                other_state = {key: [value.detach().clone().requires_grad_() for value in values]
                               for key, values in state.items()}
                retrieved = memory.retrieve(inputs, state)
                read, returned_state = reference(other_inputs, state=other_state, update=False,
                                                  update_cache=False)
                self.assert_close(retrieved, read)
                for key in state:
                    for original, returned in zip(other_state[key], returned_state[key]):
                        self.assert_close(original, returned)
                leaves = (inputs, *memory.parameters(),
                          *(value for values in state.values() for value in values))
                other_leaves = (other_inputs, *reference.parameters(),
                                *(value for values in other_state.values() for value in values))
                gradients = torch.autograd.grad(objective(retrieved), leaves, allow_unused=True)
                other_gradients = torch.autograd.grad(objective(read), other_leaves, allow_unused=True)
                for observed, expected in zip(gradients, other_gradients):
                    if observed is None or expected is None:
                        self.assertIs(observed, None)
                        self.assertIs(expected, None)
                    else:
                        self.assert_close(observed, expected)
                gate_index = 1 + list(dict(memory.named_parameters())).index("gate_proj.weight")
                self.assertIsNotNone(gradients[gate_index])
                self.assertGreater(gradients[gate_index].abs().max().item(), 1e-7)

    def test_checkpointing_preserves_every_trainable_outer_gradient(self):
        ids = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]])
        for variant in ("mag", "mac"):
            with self.subTest(variant=variant):
                model = self.make_model(variant, convolution=True).train()
                reference = copy.deepcopy(model)
                model.activation_checkpointing_enable(every_n_layers=1)
                actual = model(ids, use_cache=False).logits
                expected = reference(ids, use_cache=False).logits
                self.assert_close(actual, expected)
                objective(actual).backward()
                objective(expected).backward()
                for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertIsNotNone(other.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())
                        self.assertGreater(parameter.grad.abs().max().item(), 0.0)
                        self.assert_close(parameter.grad, other.grad)


if __name__ == "__main__":
    unittest.main()
