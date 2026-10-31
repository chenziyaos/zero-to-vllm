"""sim_so_lora.py 的回归测试：python3 -m unittest test_sim_so_lora -v"""
import random
import unittest

import sim_so_lora as S


class TestBitmask(unittest.TestCase):
    def test_columns(self):
        self.assertEqual(S.mask_cols(151936), 4748)
        self.assertEqual(S.mask_cols(128256), 4008)
        self.assertEqual(S.mask_cols(33), 2)

    def test_bytes_scale_with_spec(self):
        self.assertEqual(S.mask_bytes(256 * 4, 151936), 4 * S.mask_bytes(256, 151936))

    def test_invalid_drafts(self):
        self.assertEqual(S.grammar_invalid_drafts(4, 2), [False, False, False, True, True])
        self.assertEqual(S.grammar_invalid_drafts(2, 2), [False, False, False])


class TestThreads(unittest.TestCase):
    def test_fill_workers(self):
        self.assertEqual([S.fill_workers(c) for c in (1, 4, 16, 64)], [1, 2, 8, 8])

    def test_compile_workers(self):
        self.assertEqual([S.compile_workers(c) for c in (1, 2, 7, 32)], [1, 1, 4, 16])

    def test_parallel_only_above_threshold_and_no_spec(self):
        self.assertEqual(S.fill_ms(128, 1.0), 128)          # 不超过 128 串行
        self.assertEqual(S.fill_ms(256, 1.0, cpu=16), 32)   # 16 批 / 8 线程 = 2 波 × 16
        self.assertEqual(S.fill_ms(256, 1.0, cpu=16, k=3), 256)

    def test_sgl_timeout(self):
        self.assertEqual(S.sgl_grammar_timeout_s(0.001), 50)
        self.assertEqual(S.sgl_grammar_timeout_s(0.02), 200)


class TestSpecDecode(unittest.TestCase):
    def test_flow_cases_match_official(self):
        for name, g, drafts, r, ended, want in S.FLOW_CASES:
            with self.subTest(name=name):
                self.assertEqual(S.flow(g(), drafts, r, ended), want)

    def test_trim(self):
        for drafts, want in S.TRIM_CASES:
            req = S.Req(S.regex_grammar(), S.MockReasoner(S.TE), False)
            self.assertEqual(len(S.Manager(k=8).trim(req, list(drafts))), want)

    def test_regex_termination(self):
        for drafts, want in S.REGEX_CASES:
            self.assertEqual(S.flow(S.regex_grammar(), drafts, None, True), want)

    def test_masked_reasoning_rows(self):
        self.assertEqual(S.flow(S.json_grammar(), (' ', ' '), S.EngineReasoner(S.Z, S.TE)), 'RRR')
        self.assertEqual(S.flow(S.regex_grammar('z'), (' ', S.Z, '1'), S.EngineReasoner(S.Z, S.TE)), 'UUCC')

    def test_parallel_path_r_then_c(self):
        m = S.Manager(k=0, parallel_threshold=0)
        req = S.Req(S.json_grammar(), S.EngineReasoner(S.Z, S.TE), False)
        self.assertEqual(m.bitmask([req], [[]])[0], 'R')
        m.accept(req, [S.TE])
        self.assertEqual(m.bitmask([req], [[]])[0], 'C')

    def test_grammar_state_restored_after_bitmask(self):
        g = S.json_grammar('{"a"')
        before = g.text
        S.flow(g, (':', ' "', 'b'), S.MockReasoner(S.TE), True)
        self.assertEqual(g.text, before)


class TestAuto(unittest.TestCase):
    def test_cases(self):
        want = ['xgrammar', 'guidance', 'guidance', 'xgrammar', 'guidance', 'guidance', 'xgrammar',
                'outlines', 'guidance', 'reject', 'outlines', 'xgrammar', 'reject']
        got = [S.auto_backend(sch, tok)[0] for _, sch, tok in S.AUTO_CASES]
        self.assertEqual(got, want)

    def test_nesting_boundary(self):
        self.assertFalse(S.json_depth_exceeds(S.nest(63)))
        self.assertTrue(S.json_depth_exceeds(S.nest(64)))

    def test_tekken_lark_goes_guidance(self):
        self.assertEqual(S.auto_backend({'type': 'string'}, 'tekken', tekken_lark=True)[0], 'guidance')


class TestLoRAMemory(unittest.TestCase):
    def test_linear_in_rank_and_slots(self):
        a = S.lora_total_bytes('qwen3-8b', 16, 1)
        self.assertEqual(S.lora_total_bytes('qwen3-8b', 64, 1), 4 * a)
        self.assertEqual(S.lora_total_bytes('qwen3-8b', 16, 8), 8 * a)

    def test_fully_sharded_saves_only_with_tp(self):
        self.assertEqual(S.lora_total_bytes('qwen3-8b', 16, 1, 1, True), S.lora_total_bytes('qwen3-8b', 16, 1))
        self.assertLess(S.lora_total_bytes('qwen3-8b', 16, 1, 4, True), S.lora_total_bytes('qwen3-8b', 16, 1, 4))


class TestLoRASched(unittest.TestCase):
    def test_validate_counts_base(self):
        self.assertTrue(S.sgl_validate({None, 1, 2, 3, 4, 5, 6, 7}))
        self.assertFalse(S.sgl_validate({None, 1, 2, 3, 4, 5, 6, 7, 8}))

    def test_validate_pinned_reservation(self):
        # 8 槽、pinned={1,2}：没带 pinned 的批最多 6 个
        self.assertTrue(S.sgl_validate({3, 4, 5, 6, 7, 8}, pinned={1, 2}))
        self.assertFalse(S.sgl_validate({3, 4, 5, 6, 7, 8, 9}, pinned={1, 2}))
        self.assertTrue(S.sgl_validate({1, 3, 4, 5, 6, 7, 8}, pinned={1, 2}))

    def test_drainer_picks_least_remaining(self):
        d = S.Drainer(threshold_steps=10, max_loras=2)
        running = [S.SReq(0, 0, 1, 100), S.SReq(1, 0, 2, 20)]
        waiting = [S.SReq(2, 0, 3, 50)]
        d.update(waiting, running, now=11)
        self.assertEqual(d.draining_for, {2: 3})
        self.assertTrue(d.can_schedule(S.SReq(3, 11, 2, 24), running))     # 24 ≤ 20 × 1.2
        self.assertFalse(d.can_schedule(S.SReq(4, 11, 2, 25), running))
        d.update(waiting, running[:1], now=12)                              # 2 跑空 → 解除
        self.assertNotIn(2, d.draining_for)

    def test_drainer_idle_below_capacity(self):
        d = S.Drainer(threshold_steps=0, max_loras=8)
        d.update([S.SReq(0, 0, 9, 10)], [S.SReq(1, 0, 1, 10)], now=100)
        self.assertEqual(d.draining_for, {})

    def test_vllm_max_loras_1_blocks_second_adapter(self):
        reqs = [S.SReq(0, 0, 1, 10), S.SReq(1, 0, 2, 10), S.SReq(2, 0, None, 10)]
        out, _ = S.run_sched(reqs, 30, 'vllm', max_loras=1)
        self.assertEqual([r.start for r in out], [0, 10, 0])

    def test_sglang_base_takes_slot(self):
        reqs = [S.SReq(0, 0, None, 10), S.SReq(1, 0, 1, 10)]
        out, _ = S.run_sched(reqs, 30, 'sglang', max_loras=1)
        self.assertEqual([r.start for r in out], [0, 10])


class TestPrefix(unittest.TestCase):
    def test_vllm_key_includes_path(self):
        self.assertNotEqual(S.vllm_extra_key(('a', '/p1')), S.vllm_extra_key(('a', '/p2')))
        self.assertEqual(S.vllm_extra_key(None), ())

    def test_sglang_uuid5_stable_uuid4_not(self):
        r1, r2 = random.Random(1), random.Random(2)
        self.assertEqual(S.sgl_lora_id(('a', '/p'), True, r1), S.sgl_lora_id(('a', '/p'), True, r2))
        self.assertNotEqual(S.sgl_lora_id(('a', '/p'), False, r1), S.sgl_lora_id(('a', '/p'), False, r2))


if __name__ == '__main__':
    unittest.main()
