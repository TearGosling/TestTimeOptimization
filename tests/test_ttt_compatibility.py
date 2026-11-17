"""Offline CPU regressions for TTT's Transformers consumer interfaces."""

import copy
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from tto import TTTConfig, TTTForCausalLM
from tto.src.models.ttt.modeling_ttt import TTTCache, apply_rotary_pos_emb


class TTTCompatibilityTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = dict(
            vocab_size=24, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, mini_batch_size=2,
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
        )
        values.update(overrides)
        return TTTConfig(**values)

    def make_model(self, **overrides):
        torch.manual_seed(1907)
        return TTTForCausalLM(self.make_config(**overrides)).eval()

    def inputs(self):
        return torch.tensor([[3, 4, 5, 6], [6, 5, 4, 3]])

    def test_masked_causal_loss_and_trainer_token_count_normalization(self):
        ids = self.inputs()
        labels = ids.clone()
        labels[0, 2] = -100
        labels[1, 1:] = -100
        for kind in ("linear", "mlp"):
            with self.subTest(kind=kind):
                model = self.make_model(ttt_layer_type=kind)
                output = model(ids, labels=labels, use_cache=False)
                shifted_logits = output.logits[:, :-1].contiguous().view(-1, model.config.vocab_size)
                shifted_labels = labels[:, 1:].contiguous().view(-1)
                expected = F.cross_entropy(shifted_logits, shifted_labels)
                torch.testing.assert_close(output.loss, expected, rtol=0, atol=0)
                count = shifted_labels.ne(-100).sum()
                denominator = count + 3
                normalized = model(
                    ids, labels=labels, num_items_in_batch=denominator, use_cache=False,
                )
                expected_sum = F.cross_entropy(shifted_logits, shifted_labels, reduction="sum")
                torch.testing.assert_close(normalized.loss, expected_sum / denominator)

    def test_explicit_shifted_labels_and_instance_loss_function(self):
        model = self.make_model()
        ids = self.inputs()
        shift_labels = torch.tensor([[4, -100, 6, 7], [5, 4, 3, -100]])
        denominator = shift_labels.ne(-100).sum() + 2
        native_loss = model.loss_function
        calls = []

        def instance_loss(**kwargs):
            calls.append(kwargs)
            return native_loss(**kwargs)

        model.loss_function = instance_loss
        output = model(
            ids, labels=ids, shift_labels=shift_labels,
            num_items_in_batch=denominator, use_cache=False,
        )
        expected = F.cross_entropy(
            output.logits.view(-1, model.config.vocab_size), shift_labels.view(-1), reduction="sum",
        ) / denominator
        torch.testing.assert_close(output.loss, expected)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["shift_labels"], shift_labels)
        self.assertIs(calls[0]["num_items_in_batch"], denominator)
        self.assertNotIn("_loss_function", self.make_model().__dict__)

    @torch.no_grad()
    def test_logits_to_keep_preserves_float32_and_tensor_parallel_head(self):
        ids = self.inputs()
        for tp in (1, 2):
            with self.subTest(tp=tp):
                model = self.make_model(pretraining_tp=tp)
                full = model(ids, use_cache=False).logits
                tail = model(ids, logits_to_keep=2, use_cache=False).logits
                indices = torch.tensor([0, 2])
                selected = model(ids, logits_to_keep=indices, use_cache=False).logits
                torch.testing.assert_close(tail, full[:, -2:])
                torch.testing.assert_close(selected, full[:, indices])
                self.assertEqual(full.dtype, torch.float32)
                half_model = copy.deepcopy(model).to(torch.bfloat16)
                self.assertEqual(half_model(ids, use_cache=False).logits.dtype, torch.float32)

    @torch.no_grad()
    def test_auto_save_load_preserves_weights_buffers_and_embedding_tying(self):
        ids = self.inputs()
        for kind in ("linear", "mlp"):
            for tied in (True, False):
                with self.subTest(kind=kind, tied=tied):
                    model = self.make_model(ttt_layer_type=kind, tie_word_embeddings=tied)
                    memory = model.model.layers[0].seq_modeling_block
                    memory.W1.add_(0.125)
                    memory.learnable_token_idx.add_(0.25)
                    model.model.norm.weight.fill_(1.2)
                    expected = model(ids, use_cache=False).logits
                    with tempfile.TemporaryDirectory() as directory:
                        model.save_pretrained(directory)
                        config = AutoConfig.from_pretrained(directory, local_files_only=True)
                        restored = AutoModelForCausalLM.from_pretrained(
                            directory, local_files_only=True,
                        ).eval()
                    self.assertIsInstance(config, TTTConfig)
                    self.assertIsInstance(restored, TTTForCausalLM)
                    expected_state = model.state_dict()
                    observed_state = restored.state_dict()
                    self.assertEqual(set(expected_state), set(observed_state))
                    for name, tensor in expected_state.items():
                        torch.testing.assert_close(observed_state[name], tensor, rtol=0, atol=0)
                    restored_memory = restored.model.layers[0].seq_modeling_block
                    torch.testing.assert_close(restored_memory.token_idx, memory.token_idx, rtol=0, atol=0)
                    torch.testing.assert_close(
                        restored_memory.rotary_emb.inv_freq, memory.rotary_emb.inv_freq, rtol=0, atol=0,
                    )
                    torch.testing.assert_close(restored(ids, use_cache=False).logits, expected, rtol=0, atol=0)
                    self.assertEqual(
                        restored.get_input_embeddings().weight.data_ptr()
                        == restored.get_output_embeddings().weight.data_ptr(), tied,
                    )

    def test_auto_constructor_respects_dtype_and_tied_embeddings(self):
        for tied in (True, False):
            with self.subTest(tied=tied):
                model = AutoModelForCausalLM.from_config(
                    self.make_config(tie_word_embeddings=tied), dtype=torch.float64,
                )
                self.assertEqual(model.get_input_embeddings().weight.dtype, torch.float64)
                self.assertEqual(
                    model.get_input_embeddings().weight.data_ptr()
                    == model.get_output_embeddings().weight.data_ptr(), tied,
                )

    @torch.no_grad()
    def test_cache_aliases_and_legacy_tuple_order(self):
        ids = self.inputs()
        for kind in ("linear", "mlp"):
            with self.subTest(kind=kind):
                model = self.make_model(ttt_layer_type=kind)
                output = model(ids, labels=ids, use_cache=True, output_hidden_states=True)
                self.assertIs(output.past_key_values, output.cache_params)
                self.assertIs(output["past_key_values"], output.cache_params)
                self.assertIs(output.get("past_key_values"), output.cache_params)
                self.assertIn("past_key_values", output)
                self.assertEqual(len(output.to_tuple()), 4)
                self.assertIs(output.to_tuple()[2], output.cache_params)
                self.assertIs(output.to_tuple()[3], output.hidden_states)
                legacy_tuple = model(
                    ids, labels=ids, use_cache=True, output_hidden_states=True, return_dict=False,
                )
                self.assertEqual(len(legacy_tuple), 4)
                self.assertIsInstance(legacy_tuple[2], TTTCache)
                torch.testing.assert_close(legacy_tuple[0], output.loss)
                torch.testing.assert_close(legacy_tuple[1], output.logits)
                native = model.model(ids, use_cache=True, output_hidden_states=True)
                self.assertIs(native.past_key_values, native.cache_params)
                self.assertEqual(len(native.to_tuple()), 3)
                self.assertIs(native.to_tuple()[1], native.cache_params)
                no_cache = model(ids, use_cache=False)
                self.assertIsNone(no_cache.past_key_values)
                self.assertNotIn("past_key_values", no_cache)
                self.assertEqual(len(no_cache.to_tuple()), 1)
                with self.assertRaisesRegex(ValueError, "same cache"):
                    model(
                        ids[:, :1], cache_params=output.cache_params,
                        past_key_values=native.past_key_values, use_cache=True,
                    )

    @torch.no_grad()
    def test_full_forward_and_cached_continuation_agree_with_both_aliases(self):
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        for kind in ("linear", "mlp"):
            with self.subTest(kind=kind):
                model = self.make_model(ttt_layer_type=kind)
                full = model(ids, use_cache=False).logits
                prefix = model(ids[:, :4], use_cache=True)
                cache = prefix.past_key_values
                self.assertEqual(cache.get_seq_length(), 4)
                for index, alias in ((4, "past_key_values"), (5, "cache_params")):
                    step = model(ids[:, index:index + 1], use_cache=True, **{alias: cache})
                    self.assertIs(step.past_key_values, cache)
                    self.assertEqual(cache.get_seq_length(), index + 1)
                    torch.testing.assert_close(step.logits, full[:, index:index + 1], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(prefix.logits, full[:, :4], rtol=2e-5, atol=2e-6)

    def test_cpu_convolution_features_preserve_gradients_and_cached_continuation(self):
        ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
        for kind in ("linear", "mlp"):
            with self.subTest(kind=kind):
                model = self.make_model(
                    ttt_layer_type=kind, share_qk=True, use_gate=True, pre_conv=True,
                    pretraining_tp=2, conv_kernel=3,
                ).train()
                uncached = copy.deepcopy(model)
                expected = uncached(ids, labels=ids, use_cache=False)
                expected.loss.backward()
                output = model(ids, labels=ids, use_cache=True)
                self.assertIsNotNone(output.past_key_values)
                output.loss.backward()
                torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
                torch.testing.assert_close(output.loss, expected.loss, rtol=0, atol=0)
                for (name, parameter), (expected_name, expected_parameter) in zip(
                    model.named_parameters(), uncached.named_parameters(),
                ):
                    self.assertEqual(name, expected_name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertIsNotNone(expected_parameter.grad, name)
                    torch.testing.assert_close(
                        parameter.grad, expected_parameter.grad, rtol=2e-5, atol=2e-6,
                    )
                block = model.model.layers[0]
                for parameter in (
                    block.conv.conv.weight, block.seq_modeling_block.conv_q.weight,
                    block.seq_modeling_block.conv_k.weight, block.seq_modeling_block.g_proj.weight,
                ):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                with torch.no_grad():
                    prefix = model(ids[:, :4], use_cache=True)
                    cache = prefix.past_key_values
                    self.assertEqual(
                        set(cache.conv_states_dic), {"pre_conv", "ttt_conv_q", "ttt_conv_k"},
                    )
                    for index in (4, 5):
                        step = model(
                            ids[:, index:index + 1], past_key_values=cache, use_cache=True,
                        )
                        torch.testing.assert_close(
                            step.logits, output.logits[:, index:index + 1], rtol=2e-5, atol=2e-6,
                        )
                    for observed, expected_states in (
                        (cache.ttt_params_dict, output.past_key_values.ttt_params_dict),
                        (cache.conv_states_dic, output.past_key_values.conv_states_dic),
                    ):
                        for name, layers in expected_states.items():
                            for layer_idx, tensor in layers.items():
                                torch.testing.assert_close(
                                    observed[name][layer_idx], tensor, rtol=2e-5, atol=2e-6,
                                )

    @torch.no_grad()
    def test_cache_reordering_preserves_every_inner_parameter_and_convolution_state(self):
        model = self.make_model(share_qk=True, pre_conv=True, conv_kernel=3)
        cache = model(self.inputs(), use_cache=True).past_key_values
        original = copy.deepcopy(cache)
        indices = torch.tensor([1, 0, 1])
        cache.reorder_cache(indices)
        self.assertEqual(cache.get_seq_length(), original.get_seq_length())
        for observed, expected in (
            (cache.ttt_params_dict, original.ttt_params_dict),
            (cache.conv_states_dic, original.conv_states_dic),
        ):
            for name, layers in expected.items():
                for layer_idx, tensor in layers.items():
                    torch.testing.assert_close(
                        observed[name][layer_idx], tensor.index_select(0, indices), rtol=0, atol=0,
                    )

    @torch.no_grad()
    def test_greedy_generate_matches_full_forward_with_and_without_cache(self):
        ids = torch.tensor([[3, 4, 5, 6]])
        for kind in ("linear", "mlp"):
            with self.subTest(kind=kind):
                model = self.make_model(ttt_layer_type=kind)
                expected = ids
                for _ in range(3):
                    next_token = model(expected, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                    expected = torch.cat((expected, next_token), dim=1)
                for use_cache in (True, False):
                    observed = model.generate(
                        ids, attention_mask=torch.ones_like(ids), max_new_tokens=3,
                        do_sample=False, use_cache=use_cache, return_dict_in_generate=True,
                    )
                    torch.testing.assert_close(observed.sequences, expected, rtol=0, atol=0)
                    if use_cache:
                        self.assertIsInstance(observed.past_key_values, TTTCache)
                        self.assertEqual(observed.past_key_values.get_seq_length(), expected.shape[1] - 1)

    def test_activation_checkpointing_preserves_actual_outer_gradients(self):
        ids = self.inputs()
        for kind in ("linear", "mlp"):
            with self.subTest(kind=kind):
                model = self.make_model(ttt_layer_type=kind).train()
                checkpointed = copy.deepcopy(model)
                checkpointed.activation_checkpointing_enable(
                    activation_checkpointing_kwargs={"use_reentrant": False},
                )
                normal_output = model(ids, labels=ids, use_cache=False)
                checkpointed_output = checkpointed(ids, labels=ids, use_cache=True)
                self.assertIsNone(checkpointed_output.past_key_values)
                normal_output.loss.backward()
                checkpointed_output.loss.backward()
                torch.testing.assert_close(checkpointed_output.loss, normal_output.loss, rtol=0, atol=0)
                expected_grads = dict(model.named_parameters())
                observed_grads = dict(checkpointed.named_parameters())
                for name, parameter in expected_grads.items():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertIsNotNone(observed_grads[name].grad, name)
                    torch.testing.assert_close(observed_grads[name].grad, parameter.grad, rtol=2e-5, atol=2e-6)
                memory = checkpointed.model.layers[0].seq_modeling_block
                self.assertGreater(memory.W1.grad.abs().sum().item(), 0)
                self.assertGreater(checkpointed.model.embed_tokens.weight.grad.abs().sum().item(), 0)

    @torch.no_grad()
    def test_inputs_embeds_and_per_instance_rotary_hook(self):
        model = self.make_model()
        other = self.make_model()
        ids = self.inputs()
        expected = model(ids, use_cache=False).logits
        embedded = model(inputs_embeds=model.get_input_embeddings()(ids), use_cache=False).logits
        torch.testing.assert_close(embedded, expected, rtol=0, atol=0)
        memory = model.model.layers[0].seq_modeling_block
        calls = []

        def instance_rotary(*args, **kwargs):
            calls.append(args)
            return apply_rotary_pos_emb(*args, **kwargs)

        memory.apply_rotary_pos_emb = instance_rotary
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, rtol=0, atol=0)
        self.assertEqual(len(calls), 1)
        self.assertIs(other.model.layers[0].seq_modeling_block.apply_rotary_pos_emb, apply_rotary_pos_emb)


if __name__ == "__main__":
    unittest.main()
