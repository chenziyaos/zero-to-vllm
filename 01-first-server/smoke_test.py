"""VLLM-01 冒烟测试：只用标准库，把一个刚起好的 vLLM 服务从健康检查到流式生成过一遍。

    python3 smoke_test.py                          # 默认 http://127.0.0.1:8000
    python3 smoke_test.py --base http://host:8000 --api-key sk-xxx

依次检查：
  1. GET  /health、/version          就绪与版本号
  2. GET  /v1/models                 served-model-name 与 max_model_len
  3. POST /v1/chat/completions       非流式，看 message.reasoning 和 usage
  4. POST /v1/chat/completions       流式，测 TTFT 和平均 ITL
  5. POST /v1/completions            同一 prompt 发两次，第二次看 prompt_tokens_details.cached_tokens
  6. GET  /metrics                   抽几条 vllm: 前缀的核心指标
任何一步失败就以非零码退出，可以直接放进部署流水线。
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

OK, FAIL = '\033[32mOK\033[0m', '\033[31mFAIL\033[0m'


class Client:
    def __init__(self, base, api_key=None, timeout=120):
        self.base, self.timeout = base.rstrip('/'), timeout
        self.h = {'Content-Type': 'application/json'}
        if api_key:
            self.h['Authorization'] = f'Bearer {api_key}'

    def req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.Request(self.base + path, data=data, headers=self.h, method=method)

    def get(self, path):
        try:
            with urllib.request.urlopen(self.req('GET', path), timeout=self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw.strip().startswith(b'{') else raw.decode())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def post(self, path, body):
        with urllib.request.urlopen(self.req('POST', path, body), timeout=self.timeout) as r:
            return json.loads(r.read())

    def sse(self, path, body):
        """逐个产出 (到达时间, 解析后的 JSON)。OpenAI 兼容 SSE，每条 `data: {...}`，以 `data: [DONE]` 结束。"""
        with urllib.request.urlopen(self.req('POST', path, body), timeout=self.timeout) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    return
                yield time.perf_counter(), json.loads(payload)


def step(name, fn):
    try:
        msg = fn()
        print(f'[{OK}] {name}' + (f'  {msg}' if msg else ''))
    except Exception as e:  # noqa: BLE001
        print(f'[{FAIL}] {name}  {type(e).__name__}: {e}')
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8000')
    ap.add_argument('--api-key')
    ap.add_argument('--model', help='默认取 /v1/models 返回的第一个')
    a = ap.parse_args()
    c = Client(a.base, a.api_key)
    ctx = {}

    def readiness():
        s1, _ = c.get('/health')
        s2, v = c.get('/version')
        assert s1 == 200, f'/health={s1}'
        return f"/health=200 version={v.get('version') if isinstance(v, dict) else v}"

    def models():
        s, j = c.get('/v1/models')
        assert s == 200, j
        m = j['data'][0]
        ctx['model'] = a.model or m['id']
        return f"model={ctx['model']} max_model_len={m.get('max_model_len')} root={m.get('root')}"

    def chat():
        j = c.post('/v1/chat/completions', {
            'model': ctx['model'], 'max_tokens': 256,
            'messages': [{'role': 'user', 'content': '用一句话解释 KV cache。'}]})
        m, u = j['choices'][0]['message'], j['usage']
        # 主干把推理段放在 message.reasoning；旧字段 reasoning_content 已弃用
        think = (m.get('reasoning') or m.get('reasoning_content') or '')[:30].replace('\n', ' ')
        return (f"finish={j['choices'][0]['finish_reason']} prompt={u['prompt_tokens']} "
                f"completion={u['completion_tokens']} reasoning[:30]={think!r} "
                f"content[:40]={(m.get('content') or '')[:40]!r}")

    def chat_stream():
        t0 = time.perf_counter()
        ts, n = [], 0
        for t, j in c.sse('/v1/chat/completions', {
                'model': ctx['model'], 'max_tokens': 128, 'stream': True,
                'stream_options': {'include_usage': True},
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': '从 1 数到 30，用空格分隔。'}]}):
            if j.get('choices') and j['choices'][0]['delta'].get('content'):
                ts.append(t)
            if j.get('usage'):
                n = j['usage']['completion_tokens']
        assert ts, '没有收到任何 content 片段'
        ttft = (ts[0] - t0) * 1e3
        itl = (ts[-1] - ts[0]) / max(len(ts) - 1, 1) * 1e3
        return f'TTFT={ttft:.1f} ms  平均 ITL={itl:.2f} ms  chunks={len(ts)} completion_tokens={n}'

    def prefix_hit():
        # 前缀缓存默认开启，按 16 token 的整块命中；prompt 要明显长于一块才看得出来。
        # cached_tokens 只有在服务端加了 --enable-prompt-tokens-details 时才会返回。
        prompt = '以下是一段用于测试前缀缓存的固定前文。' * 40 + '\n问：前文重复了几次？'
        body = {'model': ctx['model'], 'prompt': prompt, 'max_tokens': 4, 'temperature': 0}
        c.post('/v1/completions', body)
        t = time.perf_counter()
        j = c.post('/v1/completions', body)
        dt = (time.perf_counter() - t) * 1e3
        u = j['usage']
        cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        return (f"prompt={u['prompt_tokens']} cached_tokens={cached}"
                f"{'（未开 --enable-prompt-tokens-details）' if cached is None else ''} 第二次耗时={dt:.0f} ms")

    def metrics():
        s, text = c.get('/metrics')
        assert s == 200, s
        keys = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc',
                'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total')
        got = {}
        for line in text.splitlines():
            for k in keys:
                if line.startswith(k + '{') or line.startswith(k + ' '):
                    got[k.split(':')[1]] = line.rsplit(' ', 1)[-1]
        assert got, '没有找到 vllm: 前缀的指标，确认 /metrics 没被网关拦截'
        return ' '.join(f'{k}={v}' for k, v in got.items())

    step('就绪检查', readiness)
    step('模型信息', models)
    step('OpenAI chat 非流式', chat)
    step('OpenAI chat 流式', chat_stream)
    step('前缀缓存命中', prefix_hit)
    step('Prometheus 指标', metrics)


if __name__ == '__main__':
    main()
