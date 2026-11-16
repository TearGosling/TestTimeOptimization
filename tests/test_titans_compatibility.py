"""Consumer-facing Transformers configuration, loading and loss regressions."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from tto import TitansConfig, TitansForCausalLM
from tto.src.models.titans.modeling_titans import apply_rotary_pos_emb


class TitansCompatibilityTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = dict(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            memory_num_heads=2, memory_head_dim=8, memory_hidden_size=12,
            memory_num_layers=2, memory_chunk_size=3, mac_segment_size=4,
            persistent_memory_tokens=2, sliding_window=3, qkv_conv_kernel=2,
            variant="mag", attn_implementation="eager",
        )
        values.update(overrides)
        return TitansConfig(**values)

    def make_model(self, **overrides):
        torch.manual_seed(2501)
        return TitansForCausalLM(self.make_config(**overrides)).eval()

    def test_legacy_rope_config_saves_canonical_parameters_and_standard_kwargs(self):
        config = self.make_config(
            rope_theta=20000.0,
            rope_scaling={"type": "linear", "factor": 2.0},
            dtype="bfloat16", return_dict=False, output_hidden_states=True,
            bos_token_id=3, eos_token_id=[4, 5], experiment_name="roundtrip",
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            saved = json.loads((Path(directory) / "config.json").read_text())
            restored = AutoConfig.from_pretrained(directory, local_files_only=True)
        self.assertIsInstance(restored, TitansConfig)
        self.assertNotIn("rope_theta", saved)
        self.assertNotIn("rope_scaling", saved)
        self.assertEqual(restored.rope_parameters["rope_theta"], 20000.0)
        self.assertEqual(restored.rope_parameters["rope_type"], "linear")
        self.assertEqual(restored.rope_parameters["factor"], 2.0)
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertFalse(restored.return_dict)
        self.assertTrue(restored.output_hidden_states)
        self.assertEqual(restored.bos_token_id, 3)
        self.assertEqual(restored.eos_token_id, [4, 5])
        self.assertEqual(restored.experiment_name, "roundtrip")

    def test_explicit_rope_parameters_take_precedence_without_mutating_input(self):
        parameters = {"rope_type": "linear", "rope_theta": 30000.0, "factor": 2.0}
        original = parameters.copy()
        config = self.make_config(rope_theta=20000.0, rope_parameters=parameters)
        self.assertEqual(parameters, original)
        self.assertIsNot(config.rope_parameters, parameters)
        self.assertEqual(config.rope_parameters, original)
        restored = TitansConfig.from_dict(config.to_dict())
        self.assertEqual(restored.rope_parameters, original)

    def test_invalid_attention_dimensions_fail_at_configuration_boundary(self):
        for overrides in (
            {"num_attention_heads": 0}, {"num_key_value_heads": 0},
            {"memory_num_heads": 0}, {"hidden_size": 0},
            {"memory_hidden_size": 0},
            {"hidden_size": 18, "num_attention_heads": 2},
            {"head_dim": 4},
            {"rope_parameters": {"partial_rotary_factor": 0.0}},
            {"rope_parameters": {"partial_rotary_factor": 0.375}},
            {"rope_parameters": {"partial_rotary_factor": 1.5}},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_config(**overrides)
        # Odd memory heads are valid when there is no rotary short-term attention.
        config = self.make_config(variant="lmm", hidden_size=18, memory_head_dim=9)
        self.assertEqual(config.hidden_size, 18)
        self.make_config(head_dim=8)
        self.make_config(head_dim=None)

    def test_standard_auto_constructor_respects_dtype_and_tied_embeddings(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                model = AutoModelForCausalLM.from_config(
                    self.make_config(tie_word_embeddings=tied),
                    dtype=torch.float64, attn_implementation="eager",
                )
                self.assertIsInstance(model, TitansForCausalLM)
                self.assertEqual(model.get_input_embeddings().weight.dtype, torch.float64)
                self.assertEqual(
                    model.get_input_embeddings().weight.data_ptr()
                    == model.get_output_embeddings().weight.data_ptr(), tied,
                )

    @torch.no_grad()
    def test_save_load_preserves_learned_state_and_reconstructs_rotary_buffers(self):
        ids = torch.tensor([[4, 5, 6, 7, 8]])
        for tied in (False, True):
            with self.subTest(tied=tied):
                model = self.make_model(
                    tie_word_embeddings=tied,
                    rope_parameters={"rope_type": "linear", "rope_theta": 20000.0, "factor": 2.0},
                )
                # Non-default learned values expose accidental reinitialization during loading.
                model.model.norm.weight.fill_(1.25)
                memory = model.model.layers[0].memory
                memory.control_proj.bias.add_(0.125)
                memory.q_conv.conv.bias.fill_(0.25)
                expected = model(ids, use_cache=False).logits
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored = AutoModelForCausalLM.from_pretrained(
                        directory, local_files_only=True, attn_implementation="eager",
                    ).eval()
                original_state = model.state_dict()
                restored_state = restored.state_dict()
                self.assertEqual(set(original_state), set(restored_state))
                for name, tensor in original_state.items():
                    torch.testing.assert_close(restored_state[name], tensor, rtol=0, atol=0)
                torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)
                self.assertEqual(restored.config.rope_parameters, model.config.rope_parameters)
                self.assertEqual(
                    restored.get_input_embeddings().weight.data_ptr()
                    == restored.get_output_embeddings().weight.data_ptr(), tied,
                )

    @torch.no_grad()
    def test_linear_rope_scales_attention_positions(self):
        default = self.make_model()
        scaled = self.make_model(rope_parameters={"rope_type": "linear", "factor": 2.0})
        x = torch.zeros(1, 3, 16)
        expected = default.model.layers[0].self_attn.rotary_emb(x, torch.tensor([[0, 2, 4]]))
        observed = scaled.model.layers[0].self_attn.rotary_emb(x, torch.tensor([[0, 4, 8]]))
        for actual, reference in zip(observed, expected):
            torch.testing.assert_close(actual, reference)

    @torch.no_grad()
    def test_partial_rope_rotates_only_the_requested_attention_dimensions(self):
        model = self.make_model(rope_parameters={"partial_rotary_factor": 0.5})
        positions = torch.tensor([[0, 2, 4]])
        query = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8)
        key = query + 0.5
        cos, sin = model.model.layers[0].self_attn.rotary_emb(query, positions)
        rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)
        for original, actual in ((query, rotated_query), (key, rotated_key)):
            prefix = original[..., :4]
            rotated_half = torch.cat([-prefix[..., 2:], prefix[..., :2]], dim=-1)
            expected = prefix * cos.unsqueeze(1) + rotated_half * sin.unsqueeze(1)
            torch.testing.assert_close(actual[..., :4], expected)
            torch.testing.assert_close(actual[..., 4:], original[..., 4:], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(model(torch.tensor([[4, 5, 6]]), use_cache=False).logits).all())

    @torch.no_grad()
    def test_rope_parameters_do_not_change_attention_free_memory(self):
        ids = torch.tensor([[4, 5, 6, 7, 8]])
        default = self.make_model(variant="lmm")
        scaled = self.make_model(variant="lmm", rope_parameters={"rope_type": "linear", "factor": 4.0})
        scaled.load_state_dict(default.state_dict())
        torch.testing.assert_close(
            default(ids, use_cache=False).logits,
            scaled(ids, use_cache=False).logits, rtol=0, atol=0,
        )

    def test_causal_loss_honors_masked_targets_and_trainer_token_normalization(self):
        model = self.make_model()
        ids = torch.tensor([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])
        labels = ids.clone()
        labels[0, 2] = -100
        labels[1, 3:] = -100
        outputs = model(ids, labels=labels, use_cache=False)
        expected_sum = F.cross_entropy(
            outputs.logits[:, :-1].reshape(-1, model.config.vocab_size).float(),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum",
        )
        expected_mean = expected_sum / labels[:, 1:].ne(-100).sum()
        torch.testing.assert_close(outputs.loss, expected_mean)
        normalized = model(
            ids, labels=labels, use_cache=False, num_items_in_batch=torch.tensor(17),
        )
        torch.testing.assert_close(normalized.loss, expected_sum / 17)
        normalized.loss.backward()
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())

    @torch.no_grad()
    def test_logits_selection_preserves_requested_positions(self):
        model = self.make_model()
        ids = torch.tensor([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])
        expected = model(ids, use_cache=False).logits
        tail = model(ids, use_cache=False, logits_to_keep=2).logits
        torch.testing.assert_close(tail, expected[:, -2:])
        positions = torch.tensor([[0, 3], [1, 4]])
        selected = model(ids, use_cache=False, logits_to_keep=positions).logits
        torch.testing.assert_close(selected, torch.take_along_dim(expected, positions[..., None], dim=1))

    @torch.no_grad()
    def test_generation_preserves_explicit_positions_across_cached_steps(self):
        ids = torch.tensor([[4, 5, 6, 7]])
        positions = torch.tensor([[2, 7, 20, 23]])
        model = self.make_model(persistent_memory_tokens=0)
        expected = ids
        current_positions = positions
        expected_logits = []
        for _ in range(3):
            logits = model(expected, position_ids=current_positions, use_cache=False).logits
            expected_logits.append(logits[:, -1])
            expected = torch.cat([expected, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
            current_positions = torch.cat([current_positions, current_positions[:, -1:] + 1], dim=1)
        observed = model.generate(
            ids, position_ids=positions, attention_mask=torch.ones_like(ids),
            max_new_tokens=3, do_sample=False, use_cache=True,
            eos_token_id=None, pad_token_id=0,
            return_dict_in_generate=True, output_logits=True,
        )
        torch.testing.assert_close(observed.sequences, expected, rtol=0, atol=0)
        for actual, reference in zip(observed.logits, expected_logits):
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)

    @torch.no_grad()
    def test_generation_continues_an_existing_memory_or_sliding_attention_cache(self):
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10]])
        generation_kwargs = dict(
            attention_mask=torch.ones_like(ids), max_new_tokens=2,
            do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0,
            return_dict_in_generate=True, output_logits=True,
        )
        for variant in ("lmm", "mag"):
            with self.subTest(variant=variant):
                model = self.make_model(variant=variant)
                expected = model.generate(ids, **generation_kwargs)
                prefix = model(ids[:, :5], use_cache=True)
                observed = model.generate(ids, past_key_values=prefix.past_key_values, **generation_kwargs)
                torch.testing.assert_close(observed.sequences, expected.sequences, rtol=0, atol=0)
                for actual, reference in zip(observed.logits, expected.logits):
                    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
