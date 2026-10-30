"""sim_prefix_cache.py 的用例：python3 -m unittest test_sim_prefix_cache -v"""
import unittest

from sim_prefix_cache import (
    NONE_HASH, RadixCache, Req, SglangCache, VllmCache, block_hashes, hash_block_tokens, plan_groups,
    reachable_mask, run, sha256_pickle, sw_blocks_held, sw_find_longest, with_group,
)


class TestBlockHash(unittest.TestCase):
    def test_chain(self):
        t = list(range(32))
        h = block_hashes(t, 16)
        self.assertEqual(h[0], hash_block_tokens(None, t[:16]))
        self.assertEqual(h[1], hash_block_tokens(h[0], t[16:]))
        self.assertEqual(NONE_HASH, sha256_pickle("vllm-none-hash"))

    def test_change_propagates_right_only(self):
        a = block_hashes(list(range(48)), 16)
        t = list(range(48))
        t[20] = -1
        b = block_hashes(t, 16)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])
        self.assertNotEqual(a[2], b[2])      # 块 2 的 token 没变，但父哈希变了

    def test_salt_and_lora(self):
        t = list(range(32))
        base = block_hashes(t, 16)
        salt = block_hashes(t, 16, cache_salt="x")
        self.assertNotEqual(base[0], salt[0])
        self.assertNotEqual(base[1], salt[1])
        # salt 只进块 0：块 1 的差异完全来自父哈希
        self.assertEqual(salt[1], hash_block_tokens(salt[0], t[16:]))
        lora = block_hashes(t, 16, lora=("a", "/p"))
        self.assertNotEqual(lora[1], hash_block_tokens(lora[0], t[16:]))

    def test_group_suffix(self):
        h = block_hashes(list(range(16)), 16)[0]
        self.assertEqual(len(with_group(h, 3)), 36)
        self.assertEqual(with_group(h, 3)[-4:], b"\x00\x00\x00\x03")


class TestVllmCache(unittest.TestCase):
    def test_hit_cap_is_n_minus_1(self):
        c = VllmCache(400, 4)
        r = Req(1, list(range(8)), [100])
        c.admit(r)
        c.release(r, True)
        # 同样 8 个 token 再来一次：上限 7，只能命中 1 块
        self.assertEqual(c.admit(Req(2, list(range(8)), [100])), 4)
        self.assertEqual(c.admit(Req(3, list(range(9)), [100])), 8)

    def test_no_dedup(self):
        c = VllmCache(40, 4)
        a, b = Req(1, list(range(6)), [6, 7, 8]), Req(2, list(range(6)), [6, 7, 8])
        for r in (a, b):
            c.admit(r)
            r.computed, r.gen = 6, 1
            for k in (7, 8):
                c.extend(r, k)
                r.computed, r.gen = k, k - 5
        self.assertEqual(c.pool.live_duplicates(), 1)
        self.assertNotEqual(c.st[1]["blocks"][1].block_id, c.st[2]["blocks"][1].block_id)

    def test_free_reversed_tail_first(self):
        c = VllmCache(4 * 10, 4, lifo_uncached=False)
        r = Req(1, list(range(12)), [1])
        c.admit(r)
        c.release(r, True)
        self.assertEqual(c.pool.q.ids()[-3:], [3, 2, 1])

    def test_doc_example_queue(self):
        c = VllmCache(4 * 10, 4, lifo_uncached=False)
        shared = list(range(1, 13))
        r0 = Req(0, shared + [13, 14, 15], [16, 17, 18])
        c.admit(r0)
        r0.computed, r0.gen = 15, 1
        for k in (16, 17):
            c.extend(r0, k)
            r0.computed, r0.gen = k, k - 14
        r1 = Req(1, shared[:10] + [50, 51, 52, 53], [60])
        self.assertEqual(c.admit(r1), 8)
        c.release(r0, True)
        c.release(r1, True)
        self.assertEqual([i - 1 for i in c.pool.q.ids()], [7, 8, 9, 4, 3, 2, 6, 5, 1, 0])
        r2 = Req(2, shared + list(range(200, 217)), [1])
        self.assertEqual(c.admit(r2), 12)
        self.assertEqual([b.block_id - 1 for b in c.st[2]["blocks"]], [0, 1, 2, 7, 8, 9, 4, 3])


class TestRadix(unittest.TestCase):
    def test_split_and_match(self):
        t = RadixCache(100)
        t.insert([1, 2, 3, 4], [10, 11, 12, 13])
        v, node = t.match([1, 2, 9])
        self.assertEqual(v, [10, 11])
        self.assertEqual(node.key, [1, 2])          # 命中停在节点中间：切成 [1,2] + [3,4]
        self.assertEqual(list(node.children), [3])

    def test_insert_reports_existing(self):
        t = RadixCache(100)
        t.insert([1, 2, 3], [10, 11, 12])
        self.assertEqual(t.insert([1, 2, 3, 4, 5], [20, 21, 22, 23, 24]), 3)
        self.assertEqual(t.match([1, 2, 3, 4, 5])[0], [10, 11, 12, 23, 24])

    def test_lru_leaf_and_lock(self):
        t = RadixCache(6)
        t.free_pages = []
        t.insert([1, 2], [0, 1])
        t.insert([5, 6], [2, 3])
        t.match([1, 2])                              # 1-2 更新访问时间
        self.assertEqual(t.evict(2), 2)
        self.assertEqual(sorted(t.free_pages), [2, 3])
        _, node = t.match([1, 2])
        t.inc_lock(node)
        self.assertEqual(t.evict(2), 0)              # 被锁住的叶子不驱逐
        t.dec_lock(node)
        self.assertEqual(t.evict(2), 2)

    def test_sglang_dedup_on_checkpoint(self):
        c = SglangCache(1000, 1)
        a, b = Req(1, list(range(50)), [1]), Req(2, list(range(50)), [1])
        c.admit(a)
        c.admit(b)
        a.computed = b.computed = 50
        c.after_step([a, b])
        self.assertEqual(c.dup_freed, 50)            # b 的 50 个 token 是重复的，插树时释放
        self.assertEqual(c.st[1]["pages"], c.st[2]["pages"])


class TestHybrid(unittest.TestCase):
    def test_sw_doc_example(self):
        cached = {2, 3, 4, 5, 11, 12, 13}
        self.assertEqual(sw_find_longest(cached, 14, 4, 1), 14)
        self.assertEqual(sw_find_longest(cached - {11, 12, 13}, 14, 4, 1), 6)
        self.assertEqual(sw_find_longest(cached - {11, 12, 13, 5}, 14, 4, 1), 5)

    def test_retention_mask(self):
        n = 32768 // 16
        self.assertIsNone(reachable_mask(0, n, 16, 128, 16, None, [32767]))
        self.assertEqual(sum(reachable_mask(0, n, 16, 128, 16, 1024, [32767])), 257)
        self.assertEqual(sum(reachable_mask(0, n, 16, 128, 16, 0, [32767])), 8)

    def test_grouping(self):
        self.assertEqual(plan_groups({"sw": 52, "full": 10})[1][0][:3], ("sw", 6, 8))
        self.assertEqual(plan_groups({"sw": 12, "full": 13})[0], 13)
        self.assertEqual(sw_blocks_held(112, 32, 16), 2)


class TestRun(unittest.TestCase):
    def test_random_no_hit(self):
        s = run(VllmCache(400000, 16), "random")
        self.assertEqual(s["hit"], 0.0)


if __name__ == "__main__":
    unittest.main()
