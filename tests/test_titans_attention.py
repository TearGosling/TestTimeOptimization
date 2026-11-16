"""Attention values and gradients against explicit Titans masks and real backends."""

import copy
import os
import unittest

import torch
import torch.nn.functional as F

from tto import TitansConfig, TitansForCausalLM
from tto.src.models.titans.modeling_titans import TitansAttention


def dense_attention_oracle(module, inputs, prefix, prefix_mask, window):
    """Independent QKV/position projection and one masked softmax per query."""
    batch, length, _ = inputs.shape
    heads, kv_heads, dim = module.num_heads, module.num_key_value_heads, module.head_dim
    query = F.silu(F.linear(inputs, module.q_proj.weight)).view(batch, length, heads, dim).transpose(1, 2)
    key = F.silu(F.linear(inputs, module.k_proj.weight)).view(batch, length, kv_heads, dim).transpose(1, 2)
    value = F.silu(F.linear(inputs, module.v_proj.weight)).view(batch, length, kv_heads, dim).transpose(1, 2)
    prefix_length = prefix.shape[1] if prefix is not None else 0

    def normalize(tensor):
        full = tensor.float()
        return (full / (full.square().sum(-1, keepdim=True) + module.config.l2_norm_eps).sqrt()).to(tensor.dtype)

    def rotate(tensor, positions):
        frequencies = module.config.rope_parameters['rope_theta'] ** (-torch.arange(0, dim, 2).float() / dim)
        angles = positions.float()[:, None] * frequencies[None]
        angles = torch.cat([angles, angles], dim=-1).to(tensor.device)
        half = dim // 2
        rotated = torch.cat([-tensor[..., half:], tensor[..., :half]], dim=-1)
        return tensor * angles.cos()[None, None] + rotated * angles.sin()[None, None]

    positions = torch.arange(length, device=inputs.device)
    if module.config.variant == 'mag':
        positions = positions + prefix_length
    query, key = rotate(normalize(query), positions), rotate(normalize(key), positions)
    if prefix is not None:
        prefix_key = F.silu(F.linear(prefix, module.k_proj.weight)).view(batch, prefix_length, kv_heads, dim).transpose(1, 2)
        prefix_value = F.silu(F.linear(prefix, module.v_proj.weight)).view(batch, prefix_length, kv_heads, dim).transpose(1, 2)
        prefix_key = rotate(normalize(prefix_key), torch.arange(prefix_length, device=inputs.device))
        key, value = torch.cat([prefix_key, key], dim=2), torch.cat([prefix_value, value], dim=2)
    groups = heads // kv_heads
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    rows = []
    for index in range(length):
        allowed = torch.zeros(batch, prefix_length + length, dtype=torch.bool, device=inputs.device)
        allowed[:, :prefix_length] = True if prefix_mask is None else prefix_mask[:, index]
        first = 0 if window is None else max(0, index - window + 1)
        allowed[:, prefix_length + first:prefix_length + index + 1] = True
        scores = (query[:, :, index:index + 1].float() @ key.float().transpose(-1, -2)) / dim ** 0.5
        scores = scores.masked_fill(~allowed[:, None, None], -torch.inf)
        rows.append(scores.softmax(-1).to(value.dtype) @ value)
    hidden = torch.cat(rows, dim=2).transpose(1, 2).reshape(batch, length, heads * dim)
    return F.linear(hidden, module.o_proj.weight)


class TitansAttentionBackendTests(unittest.TestCase):
    def config(self, variant='mal', persistent=0, implementation='eager'):
        return TitansConfig(vocab_size=32, hidden_size=32, intermediate_size=48,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            memory_num_heads=4, memory_head_dim=8, memory_hidden_size=12,
                            memory_chunk_size=3, mac_segment_size=4,
                            persistent_memory_tokens=persistent, sliding_window=3,
                            use_attention_convolution=False, use_memory_convolution=False,
                            variant=variant, attn_implementation=implementation)

    def test_eager_and_sdpa_match_explicit_window_and_prefix_oracle_gradients(self):
        torch.manual_seed(7801)
        for variant, prefix_length, window, query_masked in (
            ('mal', 0, 3, False), ('mag', 2, 3, False), ('mag', 2, None, False), ('mac', 9, None, True)
        ):
            config = self.config(variant, prefix_length)
            template = TitansAttention(config, 0).double().eval()
            for implementation in ('eager', 'sdpa'):
                with self.subTest(variant=variant, backend=implementation, window=window):
                    module = copy.deepcopy(template)
                    module.config._attn_implementation = implementation
                    inputs = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
                    prefix = torch.randn(2, prefix_length, 32, dtype=torch.float64, requires_grad=True) if prefix_length else None
                    mask = None
                    if query_masked:
                        mask = F.pad(torch.ones(7, 7, dtype=torch.bool).tril(), (2, 0), value=True)[None].expand(2, -1, -1)
                    actual = module(inputs, prefix_states=prefix, prefix_attention_mask=mask,
                                    sliding_window=window)[0]
                    expected = dense_attention_oracle(module, inputs, prefix, mask, window)
                    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
                    probe = torch.sin(torch.arange(actual.numel(), dtype=actual.dtype)).reshape_as(actual)
                    leaves = [inputs] + ([prefix] if prefix is not None else []) + list(module.parameters())
                    observed = torch.autograd.grad((actual * probe).sum(), leaves)
                    reference = torch.autograd.grad((expected * probe).sum(), leaves)
                    for a, b in zip(observed, reference):
                        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)

    @torch.no_grad()
    def test_runtime_backend_switch_and_attention_maps_preserve_logits(self):
        torch.manual_seed(7802)
        for variant in ('mag', 'mal', 'mac'):
            with self.subTest(variant=variant):
                model = TitansForCausalLM(self.config(variant, 2)).eval()
                ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
                expected = model(ids, use_cache=False, output_attentions=True)
                model.set_attn_implementation('sdpa')
                observed = model(ids, use_cache=False, output_attentions=True)
                torch.testing.assert_close(observed.logits, expected.logits)
                for a, b in zip(observed.attentions, expected.attentions):
                    if isinstance(a, tuple):
                        for left, right in zip(a, b):
                            torch.testing.assert_close(left, right)
                    else:
                        torch.testing.assert_close(a, b)

    def test_packed_attention_cannot_silently_cross_memory_streams(self):
        model = TitansForCausalLM(self.config())
        with self.assertRaisesRegex(ValueError, 'packed attention metadata'):
            model(torch.tensor([[4, 5, 6, 7]]), cu_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32))

    @unittest.skipUnless(os.environ.get('TITANS_TEST_FLASH_ATTENTION') == '1' and torch.cuda.is_available(),
                         'Enable TITANS_TEST_FLASH_ATTENTION=1 for real CUDA FlashAttention checks')
    def test_real_flash_attention_window_values_gradients_and_cached_reads(self):
        torch.manual_seed(7803)
        # Includes a query longer than the window and queries whose windows start in the cache.
        for variant, persistent in (('mal', 2), ('mag', 0), ('mag', 2), ('mac', 2)):
            with self.subTest(variant=variant, persistent=persistent):
                reference = TitansForCausalLM(self.config(variant, persistent)).cuda().to(torch.bfloat16).eval()
                observed = copy.deepcopy(reference)
                precision_reference = copy.deepcopy(reference).float()
                observed.set_attn_implementation('flash_attention_2')
                ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19, 20, 21]], device='cuda')
                expected = reference(ids, use_cache=False).logits
                actual = observed(ids, use_cache=False).logits
                precise = precision_reference(ids, use_cache=False).logits
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
                probe = torch.sin(torch.arange(actual.numel(), device='cuda')).reshape_as(actual)
                ((actual.float() * probe).sum()).backward()
                ((expected.float() * probe).sum()).backward()
                (precise * probe).sum().backward()
                # BF16 reductions can cancel near zero. Bound backend error by
                # eager BF16's measured error against the same model in FP32.
                for parameter, expected_parameter, precise_parameter in zip(
                    observed.parameters(), reference.parameters(), precision_reference.parameters()
                ):
                    if expected_parameter.grad is not None:
                        exact = precise_parameter.grad
                        eager_error = (expected_parameter.grad.float() - exact).abs().max()
                        backend_error = (parameter.grad.float() - exact).abs().max()
                        self.assertLessEqual(backend_error.item(), 2.0 * eager_error.item() + 1e-5)
                with torch.no_grad():
                    for sizes in ((1,) * 9, (2, 5, 2)):
                        cache, start, pieces = None, 0, []
                        for size in sizes:
                            result = observed(ids[:, start:start + size], past_key_values=cache, use_cache=True)
                            cache = result.past_key_values
                            pieces.append(result.logits)
                            start += size
                        torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=2e-2, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
