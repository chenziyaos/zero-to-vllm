"""sim_block_pool.py 的用例：python3 -m unittest test_sim_block_pool -v"""
import unittest

from sim_block_pool import (
    PAD_SLOT_ID, BlockPool, KVCacheManager, Req, block_table_width, make_requests, run, slot_mapping,
)


class TestBlockPool(unittest.TestCase):
    def test_null_block(self):
        pool = BlockPool(8, enable_caching=False)
        self.assertEqual(pool.null_block.block_id, 0)
        self.assertEqual(pool.free_block_queue.ids(), list(range(1, 8)))
        # null block 出现在释放列表里会被跳过，ref_cnt 不变
        pool.free_blocks([pool.null_block])
        self.assertEqual(pool.get_num_free_blocks(), 7)

    def test_lifo_reuse_without_hash(self):
        pool = BlockPool(8, enable_caching=False)
        a = pool.get_new_blocks(3)                     # 1 2 3
        pool.free_blocks(reversed(a))                  # 3 2 1 进队头
        self.assertEqual(pool.free_block_queue.ids(), [3, 2, 1, 4, 5, 6, 7])
        self.assertEqual(pool.get_new_blocks(1)[0].block_id, 3)

    def test_cached_blocks_go_to_tail(self):
        pool = BlockPool(8, enable_caching=True)
        a = pool.get_new_blocks(3)
        a[0].block_hash, a[1].block_hash = "h0", "h1"
        pool.cached.update(h0=a[0], h1=a[1])
        pool.free_blocks(reversed(a))
        # 无哈希的块 3 进队头；h1、h0 依次进队尾
        self.assertEqual(pool.free_block_queue.ids(), [3, 4, 5, 6, 7, 2, 1])

    def test_touch_removes_from_middle(self):
        pool = BlockPool(8, enable_caching=True)
        a = pool.get_new_blocks(2)
        a[0].block_hash = "h0"
        pool.free_blocks(a)
        before = pool.get_num_free_blocks()
        pool.touch([a[0]])
        self.assertEqual(a[0].ref_cnt, 1)
        self.assertNotIn(a[0].block_id, pool.free_block_queue.ids())
        self.assertEqual(pool.get_num_free_blocks(), before - 1)

    def test_evict_on_reuse(self):
        pool = BlockPool(4, enable_caching=True)
        a = pool.get_new_blocks(3)
        for i, b in enumerate(a):
            b.block_hash = f"h{i}"
            pool.cached[b.block_hash] = b
        pool.free_blocks(reversed(a))                  # 队列 h2 h1 h0
        pool.get_new_blocks(2)
        self.assertEqual(pool.evicted, ["h2", "h1"])
        self.assertIn("h0", pool.cached)

    def test_not_enough(self):
        pool = BlockPool(4, enable_caching=False)
        with self.assertRaises(ValueError):
            pool.get_new_blocks(4)                     # 只有 3 块可分


class TestManager(unittest.TestCase):
    def test_allocate_by_block(self):
        mgr = KVCacheManager(10, 4, 64)
        r = Req(0, 10, 4)
        self.assertEqual(len(mgr.allocate_slots(r, 10)), 3)
        r.computed = 10
        self.assertEqual(mgr.allocate_slots(r, 1), [])      # 位置 10、11 还在第 3 块
        r.computed = 12
        self.assertEqual(len(mgr.allocate_slots(r, 1)), 1)  # 位置 12 进第 4 块

    def test_watermark_only_for_waiting(self):
        mgr = KVCacheManager(11, 4, 64, watermark=0.2)       # 10 块可用，watermark 2 块
        w = Req(0, 36, 1)                                    # 要 9 块
        self.assertIsNone(mgr.allocate_slots(w, 36, has_scheduled_reqs=True))
        self.assertIsNotNone(mgr.allocate_slots(w, 36, has_scheduled_reqs=False))
        mgr.free(w)
        r = Req(1, 36, 1)
        r.status = "RUNNING"
        self.assertIsNotNone(mgr.allocate_slots(r, 36, has_scheduled_reqs=True))

    def test_full_sequence_must_fit(self):
        mgr = KVCacheManager(5, 4, 64)                       # 4 块
        r = Req(0, 20, 1)
        self.assertIsNone(mgr.allocate_slots(r, 8, full_sequence_must_fit=True))
        self.assertIsNotNone(mgr.allocate_slots(r, 8))


class TestBlockTable(unittest.TestCase):
    def test_width_alignment(self):
        self.assertEqual(block_table_width(2560, 16), 2560)
        self.assertEqual(block_table_width(129, 16), 136)    # 8 块 = 128 token
        self.assertEqual(block_table_width(5, 128), 5)
        self.assertEqual(block_table_width(5, 48), 8)        # gcd(128,48)=16 → 8 块对齐

    def test_slot_mapping_padding(self):
        sm = slot_mapping([[1, 2, 3, 6], [4, 5, 7]], [(12, 13), (8, 9)], 4, pad_to=4)
        self.assertEqual(sm, [24, 28, PAD_SLOT_ID, PAD_SLOT_ID])

    def test_slot_mapping_prefill(self):
        self.assertEqual(slot_mapping([[3, 9]], [(2, 6)], 4), [14, 15, 36, 37])


class TestRun(unittest.TestCase):
    def test_paged_beats_orca(self):
        reqs = make_requests("sharegpt", 200, 2048)
        p = run(reqs, "paged", 15696, 2048)
        o = run(reqs, "orca-oracle", 15696, 2048)
        m = run(reqs, "orca-max", 15696, 2048)
        self.assertGreater(p["batch"], o["batch"])
        self.assertGreater(o["batch"], m["batch"])
        self.assertLess(p["frag"] / p["cap"], 0.04)
        self.assertEqual(p["tokens"], m["tokens"])


if __name__ == "__main__":
    unittest.main()
