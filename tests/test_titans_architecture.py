"""Consumer-visible causality, streaming and padding regressions for Titans."""

import unittest

import torch

from tto import TitansConfig, TitansForCausalLM


class TitansArchitectureTests(unittest.TestCase):
    def make_model(self, variant, *, parallel=True, persistent=2):
        torch.manual_seed(2401)
        config = TitansConfig(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            memory_num_heads=2, memory_head_dim=8, memory_hidden_size=12,
            memory_num_layers=2, memory_chunk_size=3, mac_segment_size=4,
            persistent_memory_tokens=persistent, sliding_window=3,
            qkv_conv_kernel=3, variant=variant,
            use_parallel_memory_training=parallel,
        )
        return TitansForCausalLM(config).eval()

    def assert_logits_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    @torch.no_grad()
    def test_future_tokens_cannot_change_earlier_logits(self):
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        for variant in ('lmm', 'mag', 'mal', 'mac'):
            model = self.make_model(variant)
            expected = model(ids, use_cache=False).logits
            for cutoff in (1, 2, 3, 5):
                with self.subTest(variant=variant, cutoff=cutoff):
                    changed = ids.clone()
                    changed[:, cutoff:] = 17
                    observed = model(changed, use_cache=False).logits
                    self.assert_logits_close(observed[:, :cutoff], expected[:, :cutoff])

    @torch.no_grad()
    def test_cached_calls_match_one_sequence_across_chunks_segments_and_windows(self):
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19, 20, 21]])
        for variant in ('lmm', 'mag', 'mal', 'mac'):
            for parallel in (True, False):
                for persistent in (0, 2):
                    model = self.make_model(variant, parallel=parallel, persistent=persistent)
                    expected = model(ids, use_cache=False).logits
                    for sizes in ((1,) * 9, (2, 5, 2), (5, 4)):
                        with self.subTest(variant=variant, parallel=parallel, persistent=persistent, sizes=sizes):
                            cache = None
                            pieces = []
                            start = 0
                            for size in sizes:
                                result = model(ids[:, start:start + size], past_key_values=cache, use_cache=True)
                                cache = result.past_key_values
                                pieces.append(result.logits)
                                start += size
                            self.assert_logits_close(torch.cat(pieces, dim=1), expected)

    @torch.no_grad()
    def test_padding_does_not_enter_memory_convolutions_or_cached_continuations(self):
        ids = torch.tensor([[0, 0, 4, 5, 6, 7], [8, 9, 10, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        mask = ids.ne(0).long()
        continuation = torch.tensor([[12, 13], [14, 15], [16, 17]])
        for variant in ('lmm', 'mag', 'mal', 'mac'):
            with self.subTest(variant=variant):
                model = self.make_model(variant)
                padded = model(ids, attention_mask=mask, use_cache=True, output_hidden_states=True,
                               output_attentions=True)
                after = model(continuation, past_key_values=padded.past_key_values, use_cache=True).logits
                for row in range(ids.shape[0]):
                    valid = ids[row][mask[row].bool()].unsqueeze(0)
                    if valid.shape[1]:
                        expected = model(valid, use_cache=False).logits
                        self.assert_logits_close(padded.logits[row, mask[row].bool()], expected[0])
                    whole = model(torch.cat([valid, continuation[row:row + 1]], dim=1), use_cache=False).logits
                    self.assert_logits_close(after[row:row + 1], whole[:, -2:])

    @torch.no_grad()
    def test_beam_reordering_copies_unfinished_learning_state(self):
        ids = torch.tensor([[4, 5], [6, 7]])
        next_ids = torch.tensor([[8, 9], [10, 11], [12, 13]])
        order = torch.tensor([1, 0, 1])
        for variant in ('lmm', 'mag', 'mal', 'mac'):
            for padded in (False, True):
                with self.subTest(variant=variant, padded=padded):
                    model = self.make_model(variant)
                    prompt_ids = torch.cat([torch.zeros((2, 1), dtype=torch.long), ids], dim=1) if padded else ids
                    mask = prompt_ids.ne(0).long()
                    prompt = model(prompt_ids, attention_mask=mask, use_cache=True)
                    prompt.past_key_values.reorder_cache(order)
                    observed = model(next_ids, past_key_values=prompt.past_key_values, use_cache=True).logits
                    expected = model(torch.cat([ids[order], next_ids], dim=1), use_cache=False).logits[:, -2:]
                    self.assert_logits_close(observed, expected)

    def test_padded_training_preserves_outer_gradients_with_checkpointing(self):
        ids = torch.tensor([[0, 4, 5, 6, 7], [8, 9, 10, 0, 0]])
        mask = ids.ne(0)
        for variant in ('lmm', 'mag', 'mal', 'mac'):
            with self.subTest(variant=variant):
                model = self.make_model(variant).train()
                model.activation_checkpointing_enable(every_n_layers=2)
                observed = model(ids, attention_mask=mask, use_cache=False).logits
                objective = observed[mask].square().sum()
                objective.backward()
                actual = [parameter.grad.detach().clone() if parameter.grad is not None else None
                          for parameter in model.parameters()]
                model.zero_grad(set_to_none=True)
                expected_objective = sum(model(ids[row, mask[row]].unsqueeze(0), use_cache=False).logits.square().sum()
                                         for row in range(ids.shape[0]))
                expected_objective.backward()
                for parameter, gradient in zip(model.parameters(), actual):
                    if gradient is None or parameter.grad is None:
                        self.assertIs(gradient, None)
                        self.assertIs(parameter.grad, None)
                    else:
                        torch.testing.assert_close(gradient, parameter.grad, rtol=2e-4, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
