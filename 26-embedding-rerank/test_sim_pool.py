"""测 sim_pool，以及 pool_client 对着本地假服务端的请求和解析：python3 test_sim_pool.py"""
import contextlib
import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_client as pc  # noqa: E402
import sim_pool as sp  # noqa: E402
from common.specs import GPUS  # noqa: E402


class Models(unittest.TestCase):
    def test_param_counts_match_config(self):
        # Qwen3-0.6B 骨干：每层注意力 1024x2048x2 + 1024x1024x2，MLP 3x1024x3072，28 层
        self.assertEqual(sp.MODELS['qwen3-emb-0.6b'].matmul_params, 28 * (2 * 1024 * 2048 + 2 * 1024 * 1024 + 3 * 1024 * 3072))
        # XLM-R large：每层 4x1024^2 + 2x1024x4096，24 层
        self.assertEqual(sp.MODELS['bge-m3'].matmul_params, 24 * (4 * 1024 * 1024 + 2 * 1024 * 4096))
        self.assertAlmostEqual(sp.MODELS['qwen3-emb-8b'].matmul_params / 1e9, 6.95, delta=0.01)

    def test_bidirectional_attention_costs_more_per_token(self):
        m = sp.MODELS['bge-m3']
        causal = sp.PoolModel(**{**m.__dict__, 'attn': 'causal'})
        self.assertGreater(m.flops_per_token(512), causal.flops_per_token(512))
        self.assertEqual(m.flops_per_token(1), 2 * m.matmul_params + 4 * 24 * 16 * 64)

    def test_vllm_defaults(self):
        self.assertEqual(sp.vllm_defaults(sp.MODELS['qwen3-emb-0.6b']), (True, True))
        self.assertEqual(sp.vllm_defaults(sp.MODELS['qwen3-reranker-0.6b']), (True, True))
        self.assertEqual(sp.vllm_defaults(sp.MODELS['bge-m3']), (False, False))
        self.assertEqual(sp.vllm_defaults(sp.MODELS['bge-reranker-v2-m3']), (False, False))
        mean = sp.PoolModel(**{**sp.MODELS['qwen3-emb-0.6b'].__dict__, 'pooling': 'MEAN'})
        self.assertEqual(sp.vllm_defaults(mean), (False, False))

    def test_launch_flags(self):
        self.assertIn('--runner pooling', sp.vllm_flags('qwen3-emb-0.6b', sp.MODELS['qwen3-emb-0.6b']))
        self.assertIn('is_original_qwen3_reranker', sp.vllm_flags('qwen3-reranker-0.6b', sp.MODELS['qwen3-reranker-0.6b']))
        self.assertEqual(sp.sglang_flags('qwen3-emb-4b', sp.MODELS['qwen3-emb-4b']), '--is-embedding')
        # Qwen3-Reranker 在 SGLang 上不能带 --is-embedding
        self.assertNotIn('--is-embedding', sp.sglang_flags('qwen3-reranker-0.6b', sp.MODELS['qwen3-reranker-0.6b']))
        self.assertIn('--attention-backend triton', sp.sglang_flags('bge-reranker-v2-m3', sp.MODELS['bge-reranker-v2-m3']))
        # bge-m3 是编码器结构，SGLang 自动切 embedding 模式，不用写 --is-embedding
        self.assertNotIn('--is-embedding', sp.sglang_flags('bge-m3', sp.MODELS['bge-m3']))


class Ledger(unittest.TestCase):
    def test_batch_budget(self):
        h100, a100, l40s = GPUS['h100-sxm'], GPUS['a100-80g'], GPUS['l40s']
        self.assertEqual(sp.vllm_batch_budget(sp.MODELS['qwen3-emb-0.6b'], h100), 8192)
        self.assertEqual(sp.vllm_batch_budget(sp.MODELS['qwen3-emb-0.6b'], a100), 2048)
        self.assertEqual(sp.vllm_batch_budget(sp.MODELS['qwen3-emb-0.6b'], l40s), 2048)
        # 不开 chunked prefill：至少放得下一条 max_model_len
        self.assertEqual(sp.vllm_batch_budget(sp.MODELS['bge-m3'], a100), 8192)

    def test_step_time_regimes(self):
        m, gpu = sp.MODELS['qwen3-emb-0.6b'], GPUS['h100-sxm']
        t, t_mem, t_cmp = sp.step_time(m, gpu, 16, 16, 0.4)
        self.assertEqual(t, t_mem)
        t, t_mem, t_cmp = sp.step_time(m, gpu, 8192, 512, 0.4)
        self.assertEqual(t, t_cmp)


class Rerank(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(sp.rerank_tokens('bi-encoder', 100, 32, 256), 32)
        self.assertEqual(sp.rerank_tokens('cross-encoder', 100, 32, 256), 100 * (32 + 256 + 4))
        full = 61 + 32 + 4 + 256 + 9
        self.assertEqual(sp.rerank_tokens('decoder', 100, 32, 256), 100 * full)
        # 共享前缀 93 个 token，按 16 一块复用 80 个
        self.assertEqual(sp.rerank_tokens('decoder+prefix', 100, 32, 256), full + 99 * (full - 80))
        self.assertEqual(sp.rerank_tokens('decoder+prefix', 1, 32, 256), full)

    def test_prefix_needs_full_block(self):
        # 前缀不满一块时一个都复用不了
        q = 16 * 4 - 61 - 1
        full = 61 + q + 4 + 10 + 9
        self.assertEqual(sp.rerank_tokens('decoder+prefix', 3, q, 10), full + 2 * (full - 48))

    def test_scenarios_print(self):
        for s in ('models', 'ledger', 'rerank', 'mrl'):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                sp.main(['--scenario', s])
            self.assertTrue(buf.getvalue())
        # 两家都要先在 config 里声明 Matryoshka，起服时各自用 override 补上
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            sp.main(['--scenario', 'mrl'])
        self.assertIn('--hf-overrides', buf.getvalue())
        self.assertIn('--json-model-override-args', buf.getvalue())


class Client(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(pc.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(pc.cosine([1, 0], [0, 2]), 0.0)
        self.assertAlmostEqual(pc.cosine([3, 4], [6, 8]), 1.0)

    def test_rank_by_embedding(self):
        q = [1.0, 0.0]
        docs = [[0.0, 1.0], [0.9, 0.1], [0.6, 0.8]]
        self.assertEqual([i for i, _ in pc.rank_by_embedding(q, docs)], [1, 2, 0])

    def test_payloads(self):
        p = pc.rerank_payload('vllm', 'm', 'q', ['a', 'b'], top_n=1)
        self.assertEqual(p, {'model': 'm', 'query': 'q', 'documents': ['a', 'b'], 'top_n': 1})
        p = pc.rerank_payload('sglang', 'm', 'q', ['a', 'b'], top_n=1)
        self.assertEqual(p['top_n'], 1)
        self.assertEqual(p['query'], 'q')

    def test_parse_rerank(self):
        vllm = {'results': [{'index': 1, 'relevance_score': 0.9}, {'index': 0, 'relevance_score': 0.1}]}
        sgl = [{'index': 1, 'score': 0.9}, {'index': 0, 'score': 0.1}]
        self.assertEqual(pc.parse_rerank(vllm), [(1, 0.9), (0, 0.1)])
        self.assertEqual(pc.parse_rerank(sgl), [(1, 0.9), (0, 0.1)])


class FakeServer(BaseHTTPRequestHandler):
    """假服务端：/v1/embeddings 按文本长度造一个二维向量，/rerank 和 /v1/rerank 按词重叠打分。"""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/v1/embeddings':
            data = [{'index': i, 'embedding': [len(t), 1.0][:body.get('dimensions') or 2]}
                    for i, t in enumerate(body['input'])]
            out = {'data': list(reversed(data))}
        else:
            q = set(body['query'])
            scores = [(i, len(q & set(d)) / len(q)) for i, d in enumerate(body['documents'])]
            if self.path == '/rerank':
                out = {'results': [{'index': i, 'relevance_score': s} for i, s in scores]}
            else:
                out = [{'index': i, 'score': s} for i, s in scores]
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class ClientAgainstFakeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(('127.0.0.1', 0), FakeServer)
        cls.url = f'http://127.0.0.1:{cls.srv.server_address[1]}'
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_embed_sorted_by_index(self):
        self.assertEqual(pc.embed(self.url, 'm', ['a', 'bbb']), [[1, 1.0], [3, 1.0]])
        self.assertEqual(pc.embed(self.url, 'm', ['a'], dimensions=1), [[1]])

    def test_rerank_both_engines(self):
        for engine in ('vllm', 'sglang'):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                pc.main(['rerank', '--url', self.url, '--model', 'm', '--engine', engine,
                         '--query', '显存不够', '--top-n', '1'])
            self.assertIn('显存', buf.getvalue().split('\n')[1])

    def test_search_and_bench(self):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            pc.main(['search', '--url', self.url, '--model', 'm', '--query', 'q', '--top-n', '2'])
            pc.main(['bench', '--url', self.url, '--model', 'm', '--n', '8', '--concurrency', '4', '--words', '3'])
            pc.main(['bench', '--url', self.url, '--model', 'm', '--n', '4', '--concurrency', '2', '--words', '3',
                     '--rerank-docs', '5', '--engine', 'sglang'])
        self.assertIn('8 条，并发 4，embedding', buf.getvalue())
        self.assertIn('4 条，并发 2，rerank，每次 5 篇', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
