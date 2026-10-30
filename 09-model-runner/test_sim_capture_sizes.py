"""sim_capture_sizes.py 的用例：python3 -m unittest test_sim_capture_sizes -v"""
import unittest

from sim_capture_sizes import (
    ALWAYS, FULL, NEVER, NONE, PIECEWISE, UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE,
    Dispatcher, compile_ranges, lora_cases, pad_to, padding_stats, dist_uniform, resolve_mode,
    sglang_decode_sizes, sglang_defaults, sglang_prefill_sizes, vllm_capture_sizes,
)


class TestSizes(unittest.TestCase):
    def test_vllm_h100_default(self):
        s = vllm_capture_sizes(1024, 8192)
        self.assertEqual(len(s), 51)
        self.assertEqual(s[:5], [1, 2, 4, 8, 16])
        self.assertEqual(s[-1], 512)
        self.assertIn(272, s)
        self.assertNotIn(264, s)

    def test_vllm_a100_same_list(self):
        self.assertEqual(vllm_capture_sizes(256, 2048), vllm_capture_sizes(1024, 8192))

    def test_blackwell_1024(self):
        s = vllm_capture_sizes(1024, 16384, blackwell=True)
        self.assertEqual((len(s), s[-1]), (83, 1024))

    def test_interactivity(self):
        s = vllm_capture_sizes(1024, 8192, performance_mode='interactivity')
        self.assertEqual(s[:32], list(range(1, 33)))
        self.assertEqual(len(s), 76)

    def test_small_seqs_cap(self):
        s = vllm_capture_sizes(64, 8192)
        self.assertEqual(s[-1], 128)          # max = 2 × max_num_seqs
        self.assertEqual(len(s), 19)

    def test_spec_multiple_of_q(self):
        s = vllm_capture_sizes(256, 8192, query_len=4)
        self.assertTrue(all(x % 4 == 0 for x in s))
        self.assertEqual(s[0], 4)

    def test_budget_truncates(self):
        s = vllm_capture_sizes(1024, 256)
        self.assertEqual(s[-1], 256)

    def test_user_sizes(self):
        s = vllm_capture_sizes(1024, 8192, user_sizes=[1, 3, 3, 9000, 7])
        self.assertEqual(s, [1, 3, 7])

    def test_sglang(self):
        self.assertEqual(len(sglang_decode_sizes(256)), 36)
        self.assertEqual(sglang_decode_sizes(256)[:6], [1, 2, 4, 8, 12, 16])
        self.assertEqual(len(sglang_decode_sizes(512)), 52)
        self.assertEqual(sglang_decode_sizes(48)[-1], 48)
        self.assertEqual(sglang_decode_sizes(10, disable_padding=True), list(range(1, 11)))
        self.assertEqual(len(sglang_prefill_sizes(8192)), 58)
        self.assertEqual(len(sglang_decode_sizes(256, spec=True)), 51)

    def test_sglang_defaults(self):
        self.assertEqual(sglang_defaults(80), (8192, 256))
        self.assertEqual(sglang_defaults(80, tp=8), (8192, 512))
        self.assertEqual(sglang_defaults(24), (4096, 48))


class TestResolve(unittest.TestCase):
    def test_full_downgrade(self):
        self.assertEqual(resolve_mode('FULL', ALWAYS)[0], 'FULL')
        self.assertEqual(resolve_mode('FULL', UNIFORM_BATCH)[0], 'FULL_AND_PIECEWISE')
        self.assertEqual(resolve_mode('FULL', UNIFORM_SINGLE_TOKEN_DECODE)[0], 'FULL_AND_PIECEWISE')
        self.assertEqual(resolve_mode('FULL', NEVER)[0], 'ERROR')

    def test_never(self):
        self.assertEqual(resolve_mode('FULL_AND_PIECEWISE', NEVER)[0], 'PIECEWISE')
        self.assertEqual(resolve_mode('FULL_AND_PIECEWISE', NEVER, vllm_compile=False, split_attention=False)[0], 'NONE')

    def test_spec(self):
        self.assertEqual(resolve_mode('FULL_AND_PIECEWISE', UNIFORM_SINGLE_TOKEN_DECODE, query_len=4)[0], 'PIECEWISE')
        self.assertEqual(resolve_mode('FULL_AND_PIECEWISE', UNIFORM_BATCH, query_len=4)[0], 'FULL_AND_PIECEWISE')

    def test_no_piecewise(self):
        self.assertEqual(resolve_mode('FULL_AND_PIECEWISE', ALWAYS, vllm_compile=False, split_attention=False)[0],
                         'FULL_DECODE_ONLY')


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.sizes = vllm_capture_sizes(1024, 8192)

    def test_fap(self):
        d = Dispatcher('FULL_AND_PIECEWISE', self.sizes, 1024)
        m, bd = d.dispatch(37, True)
        self.assertEqual((m, bd.num_tokens, bd.num_reqs), (FULL, 40, 40))
        m, bd = d.dispatch(37, False)
        self.assertEqual((m, bd.num_tokens, bd.num_reqs), (PIECEWISE, 40, None))
        self.assertEqual(d.dispatch(600, True)[0], NONE)
        self.assertEqual(d.dispatch(512, True)[1].num_tokens, 512)

    def test_fdo_mixed_eager(self):
        d = Dispatcher('FULL_DECODE_ONLY', self.sizes, 1024)
        self.assertEqual(d.dispatch(37, True)[0], FULL)
        self.assertEqual(d.dispatch(37, False)[0], NONE)

    def test_full_mixed(self):
        d = Dispatcher('FULL', self.sizes, 1024)
        self.assertEqual(d.dispatch(300, False)[0], FULL)

    def test_none(self):
        self.assertEqual(Dispatcher('NONE', self.sizes, 1024).dispatch(1, True)[0], NONE)

    def test_lora(self):
        self.assertEqual(lora_cases(None), [0])
        self.assertEqual(lora_cases(4), [0, 5])
        self.assertEqual(lora_cases(4, specialize_active=True), [0, 1, 2, 4, 5])
        self.assertEqual(lora_cases(4, specialize_lora=False), [5])
        d = Dispatcher('FULL_AND_PIECEWISE', self.sizes, 1024, max_loras=4, specialize_active=True)
        m, bd = d.dispatch(37, True, num_active_loras=3)
        self.assertEqual((m, bd.num_active_loras), (FULL, 4))


class TestMisc(unittest.TestCase):
    def test_pad(self):
        self.assertEqual(pad_to([1, 2, 4, 8], 5), 8)
        self.assertIsNone(pad_to([1, 2, 4, 8], 9))

    def test_padding_zero_when_dense(self):
        waste, cov, _ = padding_stats(list(range(1, 33)), dist_uniform(1, 32))
        self.assertEqual((waste, cov), (0.0, 1.0))

    def test_compile_ranges(self):
        self.assertEqual(compile_ranges([512, 8192]), [(1, 512), (513, 8192)])


if __name__ == '__main__':
    unittest.main()
