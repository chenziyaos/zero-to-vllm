"""sim_parallel.py 的用例：python3 -m unittest test_sim_parallel -v"""
import unittest

from sim_parallel import (
    ARCHS, Plan, balancedness, build_groups, dp_step, dp_wave, expected_ranks_hit, gpu_loads, kv_bytes_per_rank,
    kv_heads_per_rank, ledger, linear_placement, pp_partition, rebalance, replicate_experts, skewed_load,
    weight_bytes_per_gpu,
)


class TestGroups(unittest.TestCase):
    def test_docstring_example(self):
        # parallel_state.py docstring：8 卡 TP2 x PP4
        g = build_groups(8, tp=2, pp=4)
        self.assertEqual(g['tp'], [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(g['pp'], [[0, 2, 4, 6], [1, 3, 5, 7]])

    def test_ep_spans_dp_and_tp(self):
        g = build_groups(8, dp=2, pp=2, tp=2, is_moe=True)
        self.assertEqual(g['ep'], [[0, 1, 4, 5], [2, 3, 6, 7]])
        self.assertNotIn('ep', build_groups(8, dp=2, tp=4))   # 稠密模型不建 EP 组

    def test_dcp_inside_tp(self):
        g = build_groups(8, dp=2, tp=4, dcp=2, is_moe=True)
        self.assertEqual(g['dcp'], [[0, 1], [2, 3], [4, 5], [6, 7]])
        for grp in g['dcp']:
            self.assertTrue(any(set(grp) <= set(t) for t in g['tp']))

    def test_every_rank_once(self):
        g = build_groups(16, dp=2, pp=2, tp=4, is_moe=True)
        for k in ('tp', 'pp', 'dp', 'ep'):
            self.assertEqual(sorted(r for grp in g[k] for r in grp), list(range(16)))


class TestPP(unittest.TestCase):
    def test_partition(self):
        self.assertEqual(pp_partition(61, 4), [15, 15, 16, 15])
        self.assertEqual(pp_partition(94, 4), [23, 24, 24, 23])
        self.assertEqual(pp_partition(80, 3), [27, 27, 26])

    def test_first_last_stage_heavier(self):
        a, p = ARCHS['llama70b'], Plan('', pp=4)
        w = [weight_bytes_per_gpu(a, p, s) for s in range(4)]
        self.assertGreater(w[0], w[1])
        self.assertAlmostEqual(w[0], w[3])


class TestMemory(unittest.TestCase):
    def test_params(self):
        self.assertAlmostEqual(ARCHS['dsv3'].total_params() / 1e9, 671, delta=2)
        self.assertAlmostEqual(ARCHS['qwen3-235b'].total_params() / 1e9, 235, delta=2)

    def test_gqa_kv_replicated_past_kv_heads(self):
        a = ARCHS['llama70b']
        self.assertEqual(kv_heads_per_rank(a, 8), 1)
        self.assertEqual(kv_heads_per_rank(a, 16), 1)   # 第 9~16 张卡复制
        self.assertEqual(kv_heads_per_rank(a, 4), 2)

    def test_mla_tp_does_not_shrink_kv(self):
        a = ARCHS['dsv3']
        self.assertEqual(kv_bytes_per_rank(a, Plan('', tp=8)), kv_bytes_per_rank(a, Plan('', tp=1)))
        self.assertEqual(kv_bytes_per_rank(a, Plan('', tp=8, dcp=8)) * 8, kv_bytes_per_rank(a, Plan('', tp=8)))

    def test_dp_ep_holds_more_tokens_for_mla(self):
        a = ARCHS['dsv3']
        tp8 = ledger(a, Plan('', tp=8), 'h200')
        dp8 = ledger(a, Plan('', dp=8, ep=True), 'h200')
        self.assertGreater(dp8['weights'], tp8['weights'])          # 注意力在每个 DP rank 上完整复制
        self.assertGreater(dp8['tokens_total'], 4 * tp8['tokens_total'])

    def test_tp2_70b_does_not_fit(self):
        self.assertEqual(ledger(ARCHS['llama70b'], Plan('', tp=2, dp=4), 'h100-sxm')['tokens_engine'], 0)


class TestEPLB(unittest.TestCase):
    def test_replicate_hottest(self):
        phy2log, cnt = replicate_experts([10, 1, 1, 1], 6)
        self.assertEqual(cnt[0], 3)

    def test_balancedness_improves(self):
        E, gpus, nodes = 256, 16, 2
        loads = [skewed_load(E, 32768, seed=11 + l) for l in range(4)]
        base = balancedness([gpu_loads(w, linear_placement(E, gpus), gpus) for w in loads])
        res = [balancedness([gpu_loads(w, rebalance(w, E + r, 8, nodes, gpus), gpus) for w in loads])
               for r in (0, gpus)]
        self.assertLess(base, res[0])
        self.assertLess(res[0], res[1])
        self.assertGreater(res[1], 0.95)

    def test_every_logical_expert_placed(self):
        w = skewed_load(256, 32768, seed=3)
        phy = rebalance(w, 288, 8, 4, 32)
        self.assertEqual(len(phy), 288)
        self.assertEqual(set(phy), set(range(256)))

    def test_node_fallback(self):
        w = skewed_load(256, 32768, seed=3)
        self.assertEqual(len(rebalance(w, 256, 8, 3, 16)), 256)   # 16 % 3 != 0 → 退回 num_nodes=1


class TestDP(unittest.TestCase):
    def test_pad_to_max_cg(self):
        final, mode = dp_step((37, 5, 0, 64))
        self.assertEqual((final, mode), ([64] * 4, 2))

    def test_eager_disables_pad(self):
        final, mode = dp_step((2048, 16, 16, 16))
        self.assertEqual((final, mode), ([2048, 16, 16, 16], 0))

    def test_wave_ends_on_sync_step(self):
        self.assertEqual(dp_wave((40, 12, 3, 33))[0], 48)
        self.assertEqual(dp_wave((1, 0, 0, 0))[0], 1)


class TestComm(unittest.TestCase):
    def test_ranks_hit(self):
        self.assertAlmostEqual(expected_ranks_hit(256, 256, 8), 8)
        self.assertLess(expected_ranks_hit(256, 8, 8), 8)


if __name__ == '__main__':
    unittest.main()
