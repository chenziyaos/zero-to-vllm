"""sim_pd.py 的回归测试：python3 -m unittest test_sim_pd -v"""
import unittest

import sim_pd as S

KiB = 1024


class TestLedger(unittest.TestCase):
    def test_kv_per_token_matches_guide(self):
        # vLLM 拆分指南：Qwen2.5-7B 56 KiB/token，Llama-3.1-70B 320 KiB/token
        self.assertEqual(S.kv_per_token(S.ALL['qwen2.5-7b']), 56 * KiB)
        self.assertEqual(S.kv_per_token(S.ALL['qwen3-32b']), 256 * KiB)
        self.assertEqual(S.kv_per_token(S.ALL['llama3.1-70b']), 320 * KiB)

    def test_descriptors_layers_x_kv_x_blocks(self):
        m = S.ALL['qwen3-32b']
        self.assertEqual(S.descriptors(m, 16384), 64 * 2 * 1024)
        # 不满一块也要一个描述符
        self.assertEqual(S.descriptors(m, 17), 64 * 2 * 2)


class TestPullScheduler(unittest.TestCase):
    D_PARAMS = dict(do_remote_prefill=True)

    def test_d_side_reads_whole_prompt(self):
        self.assertEqual(S.pull_matched(self.D_PARAMS, 8192, 0), (8192, True))

    def test_d_side_local_prefix_reads_only_gap(self):
        self.assertEqual(S.pull_matched(self.D_PARAMS, 8192, 4096), (4096, True))

    def test_p_side_first_round_computes_locally(self):
        self.assertEqual(S.pull_matched(dict(do_remote_decode=True), 8192, 0), (0, False))

    def test_bidirectional_threshold(self):
        p = dict(do_remote_decode=True, remote_block_ids=[1, 2], remote_engine_id='d', remote_request_id='r',
                 remote_host='h', remote_port=1, remote_num_tokens=9000)
        self.assertEqual(S.pull_matched(p, 9500, 0), (9000, True))
        self.assertEqual(S.pull_matched(p, 9500, 8960), (0, False))     # 差 40 < 64：重算
        self.assertEqual(S.pull_matched(p, 9500, 8936), (64, True))     # 正好 64：读
        self.assertEqual(S.pull_matched(p, 9500, 8900, threshold=0), (100, True))

    def test_bidirectional_needs_full_coordinates(self):
        p = dict(do_remote_decode=True, remote_block_ids=[1], remote_num_tokens=9000)
        self.assertEqual(S.pull_matched(p, 9500, 0), (0, False))

    def test_after_recv_backs_off_one_token(self):
        self.assertEqual(S.after_recv(8192, 8192), 8191)
        self.assertEqual(S.after_recv(4096, 8192), 4096)

    def test_backoff(self):
        self.assertEqual(S.prefill_backoff(), 0)
        self.assertEqual(S.prefill_backoff(has_mamba=True), 1)
        self.assertEqual(S.prefill_backoff(bounded_replay=True), 1)
        self.assertEqual(S.prefill_backoff(lookahead=3), 2)
        self.assertEqual(S.remote_prefill_count(8192, 2), 8190)

    def test_request_finished_branches(self):
        self.assertEqual(S.request_finished(dict(do_remote_decode=True), True, True, now=10), (True, 40))
        self.assertEqual(S.request_finished(dict(do_remote_decode=True), False, True), (False, None))
        self.assertEqual(S.request_finished(dict(do_remote_prefill=True), True, True), (False, None))
        self.assertEqual(S.request_finished(dict(remote_engine_id='p'), True, True), (False, None))
        self.assertEqual(S.request_finished(dict(remote_engine_id='p'), True, True, bidirectional=True),
                         (True, 480.0))


class TestLease(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual((S.LEASE, S.HB_INTERVAL, S.LEASE_EXT), (30, 5, 20))

    def test_read_while_renewed(self):
        self.assertEqual(S.lease_release_time(read_at=25), 25)
        self.assertEqual(S.lease_release_time(read_at=200), 200)

    def test_dead_d_bounded_by_lease(self):
        # 心跳把到期推到 max(30, t+20)，D 死后最多再撑 20 s
        self.assertEqual(S.lease_release_time(d_dead_at=3), 30)
        self.assertEqual(S.lease_release_time(d_dead_at=47), 65)
        self.assertEqual(S.lease_release_time(d_dead_at=0), 30)


class TestTpMapping(unittest.TestCase):
    def test_equal_tp(self):
        self.assertEqual([S.tp_sources(r, 8, 8, 8) for r in range(8)], [[r] for r in range(8)])

    def test_p_larger_dedup_heads(self):
        self.assertEqual(S.tp_sources(0, 4, 8, 8), [0, 1])
        self.assertEqual(S.tp_sources(0, 2, 8, 4), [0, 2])    # 4 个头 8 个 rank：两两重复

    def test_d_larger_head_offset(self):
        self.assertEqual(S.tp_sources(3, 8, 4, 8), [1])
        self.assertEqual(S.head_offset(3, 8, 4, 8), 1)
        self.assertEqual(S.head_offset(2, 8, 2, 4), 1)

    def test_mla_single_source(self):
        self.assertEqual(S.tp_sources(1, 4, 8, 128, mla=True), [2])


class TestSGLang(unittest.TestCase):
    def test_admission(self):
        self.assertEqual(S.sgl_admission(16000, 0, 8192, 0), 20096)
        self.assertEqual(S.sgl_admission(100, 0, 10, 2000), 2512)

    def test_dp_rank(self):
        self.assertEqual(S.sgl_dp_rank(13, 4), 1)
        self.assertEqual(S.sgl_dp_rank(13, 1), 0)


class TestSim(unittest.TestCase):
    def test_pd_removes_itl_tail(self):
        box = S.l40s_box()
        m = S.ALL['qwen2.5-7b']
        reqs = S.poisson(0.4, 40, 8192, 256, seed=1)
        co = S.summarize(S.run_colocated(box, reqs))
        pd = S.summarize(S.run_pd(box, reqs, S.xfer_fixed(m, 8192)))
        self.assertGreater(co['itl99'], 0.1)
        self.assertLess(pd['itl99'], 0.05)


if __name__ == '__main__':
    unittest.main()
