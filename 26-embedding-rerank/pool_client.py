"""Embedding / rerank 服务的最小客户端，只用标准库。服务端起法见 serve_pool.sh。

    python3 pool_client.py embed  --url http://localhost:8000 --model Qwen/Qwen3-Embedding-0.6B --text "你好" [--dimensions 256]
    python3 pool_client.py search --url http://localhost:8000 --model Qwen/Qwen3-Embedding-0.6B --query "显存不够怎么办" --docs data/docs.txt
    python3 pool_client.py rerank --url http://localhost:8001 --model BAAI/bge-reranker-v2-m3 --query "显存不够怎么办" --docs data/docs.txt --engine vllm
    python3 pool_client.py bench  --url http://localhost:8000 --model Qwen/Qwen3-Embedding-0.6B --n 400 --concurrency 32 --words 300
    python3 pool_client.py bench  --url http://localhost:30000 --model BAAI/bge-reranker-v2-m3 --rerank-docs 100 --engine sglang

search 是 bi-encoder 召回：query 和每篇文档各算一个向量，按余弦相似度排序。
rerank 是 cross-encoder 精排：把 (query, 文档) 成对交给 /rerank（vLLM）或 /v1/rerank（SGLang）打分。
bench 用线程池并发打 /v1/embeddings（加 --rerank-docs 时改打 rerank，每次带这么多篇文档），
报端到端延迟的 p50 / p99 和每秒请求数；没有 TTFT、TPOT，因为没有 decode。SGLang 没有现成的 rerank 压测脚本，用这个凑。
"""
import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(base, model, texts, dimensions=None):
    payload = {'model': model, 'input': texts}
    if dimensions:
        payload['dimensions'] = dimensions
    out = post(f'{base}/v1/embeddings', payload)
    return [d['embedding'] for d in sorted(out['data'], key=lambda d: d['index'])]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def rank_by_embedding(q, docs):
    return sorted(((i, cosine(q, d)) for i, d in enumerate(docs)), key=lambda t: -t[1])


def rerank_payload(engine, model, query, docs, top_n=None):
    """两家的请求体字段一样（query、documents、top_n）；路径不同：vLLM /rerank，SGLang /v1/rerank。"""
    p = {'model': model, 'query': query, 'documents': docs}
    if top_n:
        p['top_n'] = top_n
    return p


def parse_rerank(out):
    """vLLM 返回 {"results": [{"index", "relevance_score"}]}；SGLang 直接返回列表 [{"index", "score"}]。"""
    rows = out['results'] if isinstance(out, dict) else out
    return [(r['index'], r.get('relevance_score', r.get('score'))) for r in rows]


def load_docs(path):
    return [line.strip() for line in open(path, encoding='utf-8') if line.strip()]


def cmd_embed(a):
    t0 = time.perf_counter()
    v = embed(a.url, a.model, [a.text], a.dimensions)[0]
    ms = (time.perf_counter() - t0) * 1e3
    print(f'维度 {len(v)}，L2 范数 {math.sqrt(sum(x * x for x in v)):.4f}，前 5 维 {[round(x, 4) for x in v[:5]]}，{ms:.1f} ms')


def cmd_search(a):
    docs = load_docs(a.docs)
    t0 = time.perf_counter()
    vecs = embed(a.url, a.model, [a.query] + docs, a.dimensions)
    ms = (time.perf_counter() - t0) * 1e3
    print(f'query：{a.query}（{len(docs)} 篇文档 + 1 个 query 一次请求，{ms:.1f} ms）')
    for i, s in rank_by_embedding(vecs[0], vecs[1:])[:a.top_n]:
        print(f'  {s:.4f}  [{i}] {docs[i]}')


def cmd_rerank(a):
    docs = load_docs(a.docs)
    path = '/rerank' if a.engine == 'vllm' else '/v1/rerank'
    t0 = time.perf_counter()
    out = post(a.url + path, rerank_payload(a.engine, a.model, a.query, docs, a.top_n))
    ms = (time.perf_counter() - t0) * 1e3
    print(f'query：{a.query}（{len(docs)} 个句对，{a.engine} {path}，{ms:.1f} ms）')
    for i, s in sorted(parse_rerank(out), key=lambda t: -t[1])[:a.top_n]:
        print(f'  {s:.4f}  [{i}] {docs[i]}')


def cmd_bench(a):
    text = ' '.join(['token'] * a.words)
    path = '/rerank' if a.engine == 'vllm' else '/v1/rerank'

    def one(i):
        t0 = time.perf_counter()
        if a.rerank_docs:
            docs = [f'{i} {j} {text}' for j in range(a.rerank_docs)]
            post(a.url + path, rerank_payload(a.engine, a.model, f'query {i}', docs))
        else:
            embed(a.url, a.model, [f'{i} {text}'])
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        lat = sorted(ex.map(one, range(a.n)))
    wall = time.perf_counter() - t0
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1e3  # noqa: E731
    kind = f'rerank，每次 {a.rerank_docs} 篇' if a.rerank_docs else 'embedding'
    print(f'{a.n} 条，并发 {a.concurrency}，{kind}，每篇约 {a.words} 个词：{a.n / wall:.1f} 请求/s')
    print(f'  端到端延迟 p50 {p(0.5):.1f} ms，p99 {p(0.99):.1f} ms，均值 {statistics.mean(lat) * 1e3:.1f} ms')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('embed', 'search', 'rerank', 'bench'):
        s = sub.add_parser(name)
        s.add_argument('--url', default='http://localhost:8000')
        s.add_argument('--model', required=True)
        if name in ('embed', 'search'):
            s.add_argument('--dimensions', type=int)
        if name == 'embed':
            s.add_argument('--text', default='从 0 开始搭一个向量检索服务')
        if name in ('search', 'rerank'):
            s.add_argument('--query', required=True)
            s.add_argument('--docs', default=os.path.join(HERE, 'data', 'docs.txt'))
            s.add_argument('--top-n', type=int, default=5)
        if name in ('rerank', 'bench'):
            s.add_argument('--engine', choices=['vllm', 'sglang'], default='vllm')
        if name == 'bench':
            s.add_argument('--rerank-docs', type=int, default=0, help='大于 0 时压 rerank，每次带这么多篇文档')
            s.add_argument('--n', type=int, default=400)
            s.add_argument('--concurrency', type=int, default=32)
            s.add_argument('--words', type=int, default=300)
    a = ap.parse_args(argv)
    {'embed': cmd_embed, 'search': cmd_search, 'rerank': cmd_rerank, 'bench': cmd_bench}[a.cmd](a)


if __name__ == '__main__':
    main()
