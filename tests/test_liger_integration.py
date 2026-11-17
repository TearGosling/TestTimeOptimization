"""Instance-local Liger integration, native loading, and real CUDA parity."""

import importlib.util
import tempfile
import unittest

import torch
import torch.nn.functional as F

from tto import TTTConfig, TTTForCausalLM, TitansConfig, TitansForCausalLM
from tto.src.models.titans import modeling_titans
from tto.src.models.ttt import modeling_ttt
from tto.src.utils import LigerKernelConfig, patch_model_with_liger


_HAS_LIGER = importlib.util.find_spec("liger_kernel") is not None
_HAS_CUDA_LIGER = _HAS_LIGER and torch.cuda.is_available()
_FAMILIES = ("titans", "ttt-linear", "ttt-mlp")
_OPTIONS = (
    "rope", "rms_norm", "layer_norm", "swiglu", "geglu", "cross_entropy",
    "fused_linear_cross_entropy",
)


def _make_model(family, *, activation="silu", tied=False, partial_rope=False):
    # Initialization and inputs originate on CPU, independently of CUDA RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7319)
        common = dict(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, hidden_act=activation,
            max_position_embeddings=64, tie_word_embeddings=tied,
            bos_token_id=1, eos_token_id=None, pad_token_id=0, use_cache=False,
        )
        if family == "titans":
            return TitansForCausalLM(TitansConfig(
                **common, num_key_value_heads=1, memory_num_heads=2,
                memory_head_dim=8, memory_hidden_size=12, memory_num_layers=2,
                memory_chunk_size=2, mac_segment_size=2,
                persistent_memory_tokens=1, sliding_window=8,
                qkv_conv_kernel=2, variant="mag", attn_implementation="eager",
                rope_parameters={"partial_rotary_factor": 0.5 if partial_rope else 1.0},
            ))
        return TTTForCausalLM(TTTConfig(
            **common, ttt_layer_type=family.removeprefix("ttt-"),
            mini_batch_size=2, share_qk=False, pre_conv=False,
            conv_kernel=2, use_gate=True,
        ))


def _batch(device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(913)
    ids = torch.randint(3, 32, (2, 6), generator=generator)
    labels = ids.clone()
    labels[0, 2] = -100
    labels[1, 1] = -100
    labels[1, -1] = -100
    return ids.to(device), labels.to(device)




def _all_options(family, *, fused=False):
    return LigerKernelConfig(
        rope=True, rms_norm=True, layer_norm=family != "titans", swiglu=True,
        cross_entropy=not fused, fused_linear_cross_entropy=fused,
    )


def _cache(output):
    if isinstance(output, modeling_ttt.TTTCausalLMOutput):
        return output.cache_params
    return output.past_key_values


def _cache_kwargs(family, cache):
    return {"past_key_values" if family == "titans" else "cache_params": cache}


class _LigerAssertions:
    def assert_tree_close(self, actual, expected, *, rtol=3e-4, atol=3e-6):
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
        else:
            self.assertIs(type(actual), type(expected))
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_close(actual[key], expected[key], rtol=rtol, atol=atol)
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for observed, reference in zip(actual, expected):
                self.assert_tree_close(observed, reference, rtol=rtol, atol=atol)
        else:
            self.assertEqual(actual, expected)

    def assert_cache_close(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertIs(type(actual), type(expected))
        # Compare every adaptive weight, momentum, convolution and KV state.
        observed = {key: value for key, value in vars(actual).items() if key != "config"}
        reference = {key: value for key, value in vars(expected).items() if key != "config"}
        self.assert_tree_close(observed, reference, rtol=5e-4, atol=5e-6)

    def assert_gradients_close(self, actual, expected, *, exact=False):
        observed = dict(actual.named_parameters())
        reference = dict(expected.named_parameters())
        self.assertEqual(observed.keys(), reference.keys())
        nonzero = set()
        for name, parameter in reference.items():
            with self.subTest(parameter=name):
                gradient = observed[name].grad
                if parameter.grad is None:
                    self.assertIsNone(gradient)
                else:
                    self.assertIsNotNone(gradient)
                    torch.testing.assert_close(
                        gradient, parameter.grad,
                        rtol=0 if exact else 3e-3, atol=0 if exact else 5e-6,
                    )
                    if parameter.grad.abs().sum().item() > 0:
                        nonzero.add(name)
        # These outer gradients exercise embeddings, both sides of the head,
        # outer gated MLPs, final normalization and the adaptive memory itself.
        for name in (
            "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ):
            self.assertIn(name, nonzero)
        memory_name = (
            "model.layers.0.memory.q_proj.weight" if isinstance(expected, TitansForCausalLM)
            else "model.layers.0.seq_modeling_block.W1"
        )
        self.assertIn(memory_name, nonzero)



class LigerConfigurationTests(_LigerAssertions, unittest.TestCase):
    def assert_rejected_patch(self, model, options, exception=ValueError):
        ids, labels = _batch()
        with torch.no_grad():
            expected = model(ids, labels=labels, use_cache=False).to_tuple()
        with self.assertRaises(exception):
            patch_model_with_liger(model, LigerKernelConfig(**options))
        with torch.no_grad():
            actual = model(ids, labels=labels, use_cache=False).to_tuple()
        self.assert_tree_close(actual, expected, rtol=0, atol=0)

    def test_invalid_combinations_and_activations_fail_before_mutation(self):
        for family in _FAMILIES:
            for activation, options in (
                ("silu", {"cross_entropy": True, "fused_linear_cross_entropy": True}),
                ("silu", {"swiglu": True, "geglu": True}),
                ("relu", {"rms_norm": True, "swiglu": True}),
                ("silu", {"rope": True, "geglu": True}),
            ):
                with self.subTest(family=family, activation=activation, options=options):
                    self.assert_rejected_patch(_make_model(family, activation=activation), options)

    def test_non_boolean_options_are_rejected_without_patching(self):
        model = _make_model("ttt-linear")
        for name in _OPTIONS:
            for value in (1, "false", None):
                with self.subTest(option=name, value=value):
                    self.assert_rejected_patch(model, {name: value}, (TypeError, ValueError))

    def test_titans_layer_norm_is_rejected_before_other_options_are_applied(self):
        model = _make_model("titans")
        self.assert_rejected_patch(model, {"rms_norm": True, "layer_norm": True})

    def test_attention_free_titans_rejects_rope_without_mutation(self):
        model = _make_model("titans")
        config = model.config.to_dict()
        config["variant"] = "lmm"
        model = TitansForCausalLM(TitansConfig(**config))
        self.assert_rejected_patch(model, {"rope": True, "rms_norm": True})

    def test_unsupported_models_and_config_objects_are_rejected(self):
        with self.assertRaises(TypeError):
            patch_model_with_liger(torch.nn.Linear(4, 4), LigerKernelConfig())
        model = _make_model("titans")
        with self.assertRaises(TypeError):
            patch_model_with_liger(model, {"rms_norm": True})

    def test_disabled_patching_preserves_outputs_caches_and_outer_gradients(self):
        ids, labels = _batch()
        for family in _FAMILIES:
            with self.subTest(family=family):
                native = _make_model(family).train()
                candidate = _make_model(family).train()
                candidate.load_state_dict(native.state_dict())
                patch_model_with_liger(candidate, LigerKernelConfig())
                expected = native(ids, labels=labels, output_hidden_states=True, use_cache=False)
                actual = candidate(ids, labels=labels, output_hidden_states=True, use_cache=False)
                self.assertIs(type(actual), type(expected))
                self.assert_tree_close(actual.to_tuple(), expected.to_tuple(), rtol=0, atol=0)
                actual.loss.backward()
                expected.loss.backward()
                self.assert_gradients_close(candidate, native, exact=True)
                with torch.no_grad():
                    expected = native.eval()(ids, use_cache=True)
                    actual = candidate.eval()(ids, use_cache=True)
                self.assert_tree_close(actual.logits, expected.logits, rtol=0, atol=0)
                self.assert_cache_close(_cache(actual), _cache(expected))


@unittest.skipUnless(_HAS_LIGER, "liger-kernel is not installed")
class LigerInstanceIsolationTests(_LigerAssertions, unittest.TestCase):
    def test_patching_does_not_change_existing_or_future_native_consumers(self):
        ids, labels = _batch()
        classifier_input = torch.tensor([[0.2, -0.3, 1.1], [0.4, 0.8, -0.7]])
        classifier_labels = torch.tensor([2, 0])
        for family in _FAMILIES:
            for fused in (False, True):
                with self.subTest(family=family, fused=fused):
                    target = _make_model(family)
                    untouched = _make_model(family).train()
                    expected = untouched(ids, labels=labels, use_cache=False)
                    expected.loss.backward()
                    expected_gradients = {
                        name: None if parameter.grad is None else parameter.grad.clone()
                        for name, parameter in untouched.named_parameters()
                    }
                    untouched.zero_grad(set_to_none=True)
                    x = classifier_input.clone().requires_grad_()
                    classifier_loss = F.cross_entropy(x, classifier_labels)
                    classifier_loss.backward()
                    classifier_gradient = x.grad.clone()

                    patch_model_with_liger(target, _all_options(family, fused=fused))
                    for consumer in (untouched, _make_model(family).train()):
                        actual = consumer(ids, labels=labels, use_cache=False)
                        self.assert_tree_close(actual.logits, expected.logits, rtol=0, atol=0)
                        self.assert_tree_close(actual.loss, expected.loss, rtol=0, atol=0)
                        actual.loss.backward()
                        for name, parameter in consumer.named_parameters():
                            self.assert_tree_close(parameter.grad, expected_gradients[name], rtol=0, atol=0)
                    x = classifier_input.clone().requires_grad_()
                    actual_loss = F.cross_entropy(x, classifier_labels)
                    actual_loss.backward()
                    self.assert_tree_close(actual_loss, classifier_loss, rtol=0, atol=0)
                    self.assert_tree_close(x.grad, classifier_gradient, rtol=0, atol=0)

    def test_patched_checkpoints_load_as_native_models_without_liger_state(self):
        ids, _ = _batch()
        for family in _FAMILIES:
            for tied in (False, True):
                with self.subTest(family=family, tied=tied):
                    model = _make_model(family, tied=tied).eval()
                    with torch.no_grad():
                        expected = model(ids, use_cache=False).logits
                    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                    original_config = model.config.to_dict()
                    patch_model_with_liger(model, _all_options(family, fused=True))
                    self.assertEqual(model.config.to_dict(), original_config)
                    with tempfile.TemporaryDirectory() as directory:
                        model.save_pretrained(directory)
                        restored = type(model).from_pretrained(
                            directory, local_files_only=True,
                            attn_implementation=model.config._attn_implementation,
                        ).eval()
                    self.assert_tree_close(restored.state_dict(), state, rtol=0, atol=0)
                    with torch.no_grad():
                        actual = restored(ids, use_cache=False).logits
                    self.assert_tree_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(
                        restored.lm_head.weight.data_ptr() == restored.model.embed_tokens.weight.data_ptr(), tied,
                    )


@unittest.skipUnless(_HAS_CUDA_LIGER, "CUDA and liger-kernel are required")
class LigerCUDAParityTests(_LigerAssertions, unittest.TestCase):
    def setUp(self):
        self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", self.old_tf32)

    def make_pair(self, family, options, *, activation="silu", partial_rope=False):
        native = _make_model(family, activation=activation, partial_rope=partial_rope)
        patched = _make_model(family, activation=activation, partial_rope=partial_rope)
        patched.load_state_dict(native.state_dict())
        patch_model_with_liger(patched, options)
        return native.cuda().train(), patched.cuda().train()

    def check_component(self, component, *, families=_FAMILIES, activation="silu", partial_rope=False):
        ids, labels = _batch("cuda")
        for family in families:
            with self.subTest(component=component, family=family):
                native, patched = self.make_pair(
                    family, LigerKernelConfig(**{component: True}),
                    activation=activation, partial_rope=partial_rope,
                )
                expected = native(ids, labels=labels, output_hidden_states=True, use_cache=False)
                actual = patched(ids, labels=labels, output_hidden_states=True, use_cache=False)
                self.assertIs(type(actual), type(expected))
                self.assert_tree_close(actual.logits, expected.logits)
                self.assert_tree_close(actual.hidden_states, expected.hidden_states)
                self.assert_tree_close(actual.loss, expected.loss)
                expected.loss.backward()
                actual.loss.backward()
                self.assert_gradients_close(patched, native)

    def test_rms_norm_forward_and_outer_backward(self):
        self.check_component("rms_norm")

    def test_ttt_post_layer_norm_forward_and_outer_backward(self):
        self.check_component("layer_norm", families=("ttt-linear", "ttt-mlp"))

    def test_swiglu_forward_and_outer_backward(self):
        self.check_component("swiglu")

    def test_geglu_forward_and_outer_backward_uses_matching_tanh_activation(self):
        self.check_component("geglu", activation="gelu_pytorch_tanh")

    def test_rope_forward_and_outer_backward_preserves_ttt_permutation(self):
        self.check_component("rope")

    def test_titans_partial_rope_forward_and_outer_backward(self):
        self.check_component("rope", families=("titans",), partial_rope=True)

    def test_partial_rope_leaves_unrotated_channels_and_their_gradients_exact(self):
        native, patched = self.make_pair("titans", LigerKernelConfig(rope=True), partial_rope=True)
        native_attention = native.model.layers[0].self_attn
        patched_attention = patched.model.layers[0].self_attn
        generator = torch.Generator(device="cpu").manual_seed(471)
        query = torch.randn(2, 2, 6, 8, generator=generator).cuda().requires_grad_()
        key = torch.randn(2, 1, 6, 8, generator=generator).cuda().requires_grad_()
        q_ref = query.detach().clone().requires_grad_()
        k_ref = key.detach().clone().requires_grad_()
        positions = torch.arange(6, device="cuda").unsqueeze(0).expand(2, -1)
        cos, sin = native_attention.rotary_emb(query, positions)
        expected = native_attention.apply_rotary_pos_emb(q_ref, k_ref, cos, sin)
        actual = patched_attention.apply_rotary_pos_emb(query, key, cos, sin)
        self.assert_tree_close(actual, expected)
        for rotated, original in zip(actual, (query, key)):
            torch.testing.assert_close(rotated[..., 4:], original[..., 4:], rtol=0, atol=0)
        sum(tensor.square().sum() for tensor in expected).backward()
        sum(tensor.square().sum() for tensor in actual).backward()
        self.assert_tree_close(query.grad, q_ref.grad)
        self.assert_tree_close(key.grad, k_ref.grad)
        torch.testing.assert_close(query.grad[..., 4:], 2 * query[..., 4:], rtol=0, atol=0)
        torch.testing.assert_close(key.grad[..., 4:], 2 * key[..., 4:], rtol=0, atol=0)

    def test_cross_entropy_masked_labels_and_outer_backward(self):
        self.check_component("cross_entropy")

    def test_loss_kernels_follow_trainer_token_count_and_explicit_shift_labels(self):
        ids, labels = _batch("cuda")
        shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        # Deliberately differs from the local token count, as in accumulation.
        num_items = shifted.ne(-100).sum() + 7
        for family in _FAMILIES:
            for component in ("cross_entropy", "fused_linear_cross_entropy"):
                for explicit_shift in (False, True):
                    with self.subTest(family=family, component=component, explicit_shift=explicit_shift):
                        native, patched = self.make_pair(family, LigerKernelConfig(**{component: True}))
                        kwargs = {"num_items_in_batch": num_items}
                        effective_labels = labels
                        if explicit_shift:
                            # Unshifted labels intentionally differ so ignoring
                            # the Trainer's pre-shifted labels cannot pass.
                            effective_labels = torch.full_like(labels, 3)
                            kwargs["shift_labels"] = shifted
                        expected_output = native(ids, use_cache=False)
                        expected = F.cross_entropy(
                            expected_output.logits.float().reshape(-1, native.config.vocab_size),
                            shifted.reshape(-1), reduction="sum", ignore_index=-100,
                        ) / num_items
                        actual = patched(ids, labels=effective_labels, use_cache=False, **kwargs)
                        self.assert_tree_close(actual.loss, expected)
                        expected.backward()
                        actual.loss.backward()
                        self.assert_gradients_close(patched, native)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices are required")
    def test_data_parallel_keeps_patched_forwards_bound_to_each_replica(self):
        ids, labels = _batch("cuda")
        for family in _FAMILIES:
            for fused in (False, True):
                with self.subTest(family=family, fused=fused):
                    native, patched = self.make_pair(family, _all_options(family, fused=fused))
                    expected = torch.nn.DataParallel(native, device_ids=[0, 1])(
                        ids, labels=labels, use_cache=False,
                    )
                    actual = torch.nn.DataParallel(patched, device_ids=[0, 1])(
                        ids, labels=labels, use_cache=False,
                    )
                    self.assert_tree_close(actual.loss, expected.loss)
                    if fused:
                        self.assertIsNone(actual.logits)
                    else:
                        self.assert_tree_close(actual.logits, expected.logits)
                    expected.loss.mean().backward()
                    actual.loss.mean().backward()
                    self.assert_gradients_close(patched, native)

    def test_fused_training_default_skips_logits_and_preserves_loss_cache_and_gradients(self):
        ids, labels = _batch("cuda")
        for family in _FAMILIES:
            with self.subTest(family=family):
                native, patched = self.make_pair(family, LigerKernelConfig(fused_linear_cross_entropy=True))
                expected = native(ids, labels=labels, use_cache=True, output_hidden_states=True)
                actual = patched(ids, labels=labels, use_cache=True, output_hidden_states=True)
                self.assertIs(type(actual), type(expected))
                self.assertIsNone(actual.logits)
                self.assert_tree_close(actual.loss, expected.loss)
                self.assert_tree_close(actual.hidden_states, expected.hidden_states)
                self.assert_cache_close(_cache(actual), _cache(expected))
                expected.loss.backward()
                actual.loss.backward()
                self.assert_gradients_close(patched, native)
                native.eval()
                patched.eval()
                next_ids = ids[:, :1]
                with torch.no_grad():
                    # Titans intentionally omits caches while training. Obtain
                    # an inference prefill before checking real continuation.
                    if _cache(expected) is None:
                        expected = native(ids, use_cache=True)
                        actual = patched(ids, use_cache=True)
                    expected_next = native(next_ids, use_cache=True, **_cache_kwargs(family, _cache(expected)))
                    actual_next = patched(next_ids, use_cache=True, **_cache_kwargs(family, _cache(actual)))
                self.assert_tree_close(actual_next.logits, expected_next.logits)
                self.assert_cache_close(_cache(actual_next), _cache(expected_next))

    def test_fused_logit_overrides_eval_defaults_and_tuple_order(self):
        ids, labels = _batch("cuda")
        for family in _FAMILIES:
            native, patched = self.make_pair(family, LigerKernelConfig(fused_linear_cross_entropy=True))
            for training, override in ((True, False), (False, None), (False, False), (False, True)):
                with self.subTest(family=family, training=training, skip_logits=override):
                    native.train(training)
                    patched.train(training)
                    kwargs = {} if override is None else {"skip_logits": override}
                    expected = native(ids, labels=labels, use_cache=False, output_hidden_states=True)
                    actual = patched(ids, labels=labels, use_cache=False, output_hidden_states=True, **kwargs)
                    self.assert_tree_close(actual.loss, expected.loss)
                    if override is True:
                        self.assertIsNone(actual.logits)
                    else:
                        self.assert_tree_close(actual.logits, expected.logits)
                    self.assertIs(type(actual), type(expected))
                    as_tuple = patched(
                        ids, labels=labels, use_cache=False, output_hidden_states=True,
                        return_dict=False, skip_logits=False,
                    )
                    self.assertIsInstance(as_tuple, tuple)
                    self.assert_tree_close(as_tuple, expected.to_tuple())
                    skipped_tuple = patched(
                        ids, labels=labels, use_cache=False, output_hidden_states=True,
                        return_dict=False, skip_logits=True,
                    )
                    self.assertEqual(len(skipped_tuple), len(as_tuple))
                    self.assert_tree_close(skipped_tuple[0], expected.loss)
                    self.assertIsNone(skipped_tuple[1])
                    self.assert_tree_close(skipped_tuple[2:], as_tuple[2:])
                    native.zero_grad(set_to_none=True)
                    patched.zero_grad(set_to_none=True)
                    expected.loss.backward()
                    actual.loss.backward()
                    self.assert_gradients_close(patched, native)
            # Without labels, even training needs logits for ordinary callers.
            patched.train()
            native.train()
            self.assert_tree_close(
                patched(ids, use_cache=False).logits, native(ids, use_cache=False).logits,
            )

    def test_combined_kernels_preserve_fused_loss_and_outer_gradients(self):
        ids, labels = _batch("cuda")
        for family in _FAMILIES:
            with self.subTest(family=family):
                native, patched = self.make_pair(family, _all_options(family, fused=True))
                expected = native(ids, labels=labels, use_cache=False)
                actual = patched(ids, labels=labels, use_cache=False)
                self.assertIsNone(actual.logits)
                self.assert_tree_close(actual.loss, expected.loss)
                expected.loss.backward()
                actual.loss.backward()
                self.assert_gradients_close(patched, native)

    def test_patched_inference_honors_logits_to_keep_and_generation(self):
        ids, _ = _batch("cuda")
        for family in _FAMILIES:
            with self.subTest(family=family):
                native, patched = self.make_pair(family, _all_options(family, fused=True))
                native.eval()
                patched.eval()
                with torch.no_grad():
                    full = native(ids, use_cache=False).logits
                    for keep in (1, 3, torch.tensor([0, 2, 5], device="cuda")):
                        actual = patched(ids, use_cache=False, logits_to_keep=keep)
                        selected = full[:, -keep:] if isinstance(keep, int) else full[:, keep]
                        self.assert_tree_close(actual.logits, selected)
                    generation_kwargs = dict(
                        max_new_tokens=3, do_sample=False, use_cache=True,
                        pad_token_id=0, eos_token_id=None,
                    )
                    expected_ids = native.generate(ids[:1], **generation_kwargs)
                    actual_ids = patched.generate(ids[:1], **generation_kwargs)
                    self.assertEqual(actual_ids.shape, (1, ids.shape[1] + 3))
                    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
