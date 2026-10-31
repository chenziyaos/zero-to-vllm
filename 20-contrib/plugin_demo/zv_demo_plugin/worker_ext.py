"""worker 扩展：通过 --worker-extension-cls 混入每个 worker，供 collective_rpc 调用。

vllm/v1/worker/worker_base.py:309-353：扩展类被追加到 worker 类的 __bases__；
扩展里任何非 dunder 属性和 worker 已有属性重名都会 assert 失败，所以方法统一加 zv_ 前缀。
"""


class ZvWorkerExtension:
    def zv_get_scheduler_config(self):
        """只回传可 JSON 序列化的标量，collective_rpc 结果要经 msgpack 回到 API server。"""
        sc = self.scheduler_config
        return {
            'rank': self.rank,
            'max_num_seqs': sc.max_num_seqs,
            'max_num_batched_tokens': sc.max_num_batched_tokens,
            'max_model_len': getattr(sc, 'max_model_len', None),
            'enable_chunked_prefill': getattr(sc, 'enable_chunked_prefill', None),
        }
