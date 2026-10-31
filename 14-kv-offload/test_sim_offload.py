"""sim_offload.py 的回归测试：python3 -m unittest test_sim_offload -v"""
import unittest

import sim_offload as S


class TestConnectorRules(unittest.TestCase):
    def test_multi_connector_first_positive_wins(self):
        self.assertEqual(S.multi_connector_pick([(0, False), (30000, True)]), (30000, True, 1))
        # 后面的命中更长也不换：顺序就是优先级
        self.assertEqual(S.multi_connector_pick([(12000, True), (60000, True)]), (12000, True, 0))
        self.assertEqual(S.multi_connector_pick([(0, False), (0, False)]), (0, False, None))

    def test_multi_connector_none_blocks_all(self):
        self.assertEqual(S.multi_connector_pick([(12000, True), (None, False)]), (None, False, None))
        self.assertEqual(S.multi_connector_pick([(None, False), (5, True)]), (None, False, None))

    def test_offload_in_flight_returns_none(self):
        self.assertEqual(S.offload_matched(True, False, 8192, 0, 9000, 1, 0, 0), (None, False, 0))

    def test_offload_skip_or_failed_gives_zero(self):
        hit, la, _ = S.offload_matched(False, True, 8192, 0, 9000, 1, 0, 0)
        self.assertEqual((hit, la), (0, False))

    def test_sync_budget(self):
        bpt = S.kv_per_card(S.MODELS['llama3.1-8b'])
        budget = 2 * S.GiB
        # 0.5 GiB 进预算 → 同步
        hit, la, used = S.offload_matched(False, False, 4096, 0, 6000, bpt, budget, 0)
        self.assertEqual((hit, la, used), (4096, False, 4096 * bpt))
        # 全命中：必须留一个 token 计算，不能走同步
        hit, la, used2 = S.offload_matched(False, False, 4096, 0, 4096, bpt, budget, used)
        self.assertTrue(la)
        self.assertEqual(used2, used)
        # 超过剩余预算 → 异步
        hit, la, _ = S.offload_matched(False, False, 81920, 0, 82242, bpt, budget, 0)
        self.assertTrue(la)
        # 预算 0（默认）→ 有命中就异步
        self.assertTrue(S.offload_matched(False, False, 16, 0, 100, bpt, 0, 0)[1])

    def test_after_recv_backs_off_one_token(self):
        self.assertEqual(S.after_recv(81920, 81920), 81919)
        self.assertEqual(S.after_recv(81920, 82242), 81920)

    def test_schedule_external_drops_partial_tail(self):
        self.assertEqual(S.schedule_external(20008, 61920), (20000, 61920))
        self.assertEqual(S.schedule_external(20000, 61920), (20000, 61920))
        self.assertEqual(S.schedule_external(20008, 0), (20008, 0))


class TestLedger(unittest.TestCase):
    def test_layout_matches_blog(self):
        frag, blk, n = S.layout_sizes(S.MODELS['llama3.1-8b'])
        self.assertEqual((frag, blk, n), (32 * 1024, 2 * 2**20, 64))
        frag, blk, _ = S.layout_sizes(S.MODELS['llama3.1-70b'], tp=4)
        self.assertEqual((frag, blk), (8 * 1024, int(1.25 * 2**20)))

    def test_kv_per_token(self):
        self.assertEqual(S.kv_per_card(S.MODELS['llama3.1-8b']), 128 * 1024)

    def test_hicache_timeout_code_vs_doc(self):
        self.assertAlmostEqual(S.hicache_timeout(81920), 21.0)
        self.assertGreater(S.hicache_timeout(10**7), 30.0)
        self.assertAlmostEqual(S.hicache_timeout(81920, 2.0, 0.1, 30.0), 10.0)
        self.assertEqual(S.hicache_timeout(10**7, 2.0, 0.1, 30.0), 30.0)


class TestPolicies(unittest.TestCase):
    def test_arc_ghost_hits_move_target(self):
        a = S.ARC(4)
        a.insert(['a', 'b', 'c', 'd'], set())
        a.insert(['e'], set())                      # 淘汰 a 进 B1
        self.assertIn('a', a.b1)
        p0 = a.p
        a.insert(['a'], set())                      # B1 ghost 命中 → target_t1_size 变大
        self.assertGreater(a.p, p0)
        a.touch(['a'])                              # 复用 → 升 T2
        self.assertIn('a', a.t2)

    def test_arc_protects_reused_from_scan(self):
        a = S.ARC(4)
        a.insert(['h1', 'h2'], set())
        a.touch(['h1', 'h2'])                       # 热块在 T2
        a.insert(['s1', 's2', 's3', 's4'], set())   # 一次性扫描
        self.assertIn('h1', a)
        self.assertIn('h2', a)
        lru = S.LRU(4)
        lru.insert(['h1', 'h2'], set())
        lru.insert(['s1', 's2', 's3', 's4'], set())
        self.assertNotIn('h1', lru)

    def test_store_threshold_skips_one_shot(self):
        t = S.Threshold(S.LRU(8), 2)
        self.assertEqual(t.insert(['x'], set()), 0)
        self.assertEqual(t.insert(['x'], set()), 1)
        self.assertIn('x', t)

    def test_small_cpu_tier_is_mirror(self):
        bpt = S.kv_per_card(S.MODELS['llama3.1-8b'])
        gpu_chunks = 457_696 // S.CHUNK
        small = int(32 * S.GiB / bpt) // S.CHUNK
        big = int(256 * S.GiB / bpt) // S.CHUNK
        st_small = S.run_tiers(32, gpu_chunks, small)
        st_big = S.run_tiers(32, gpu_chunks, big)
        self.assertLess(st_small['cpu'] / st_small['total'], 0.01)
        self.assertGreater(st_big['cpu'] / st_big['total'], 0.5)


if __name__ == '__main__':
    unittest.main()
