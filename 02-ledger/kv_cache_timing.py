"""KV cache 开关对照：同一个小模型、同一段 prompt，带 cache 和不带 cache 各生成一遍，比时间和运算量。

    python3 kv_cache_timing.py                       # 默认：4 层、d=128、prompt 32、生成 96
    python3 kv_cache_timing.py --prompt 128 --gen 256
    python3 kv_cache_timing.py --params-b 8 --layers 36 --d-model 4096 --prompt 1024 --gen 512   # 只算账，不跑

模型是随机权重的 decoder-only Transformer（RMSNorm、多头因果注意力、两层 MLP），用 float64 和 greedy 解码，
所以两种做法生成的 token 必须逐个相同，脚本会检查这一点。

不带 cache：每生成一个 token，都把「prompt + 已生成」整段重新过一遍模型，只取最后一个位置的 logits。
带 cache：先 prefill 整段 prompt，把每层的 K、V 存下来；之后每步只算新 token 一行，K、V 追加到缓存里。

运算量按矩阵乘的乘加次数算（1 次乘加 = 2 FLOP）：
  每 token 的投影和 MLP  2 x 层数 x (4 x d^2 + 2 x d x d_ff) + 2 x d x vocab（只算最后一个位置的 logits）
  注意力                 2 x 层数 x 2 x d x 可见长度（QK^T 和 PV 各一次）
--params-b 模式把前一项换成 2 x 参数量，按同样的公式给出真实模型尺寸下的账。
"""
import argparse
import time

import numpy as np


def init(layers, d, heads, d_ff, vocab, seed=0):
    rng = np.random.default_rng(seed)
    s = 1 / np.sqrt(d)
    w = {'emb': rng.normal(0, 1, (vocab, d)), 'out': rng.normal(0, s, (d, vocab)), 'layers': []}
    for _ in range(layers):
        w['layers'].append({
            'qkv': rng.normal(0, s, (d, 3 * d)), 'o': rng.normal(0, s, (d, d)),
            'up': rng.normal(0, s, (d, d_ff)), 'down': rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d)),
        })
    w['heads'] = heads
    return w


def rms(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def attend(q, k, v, heads, q_pos0):
    """q: [t, d]，k/v: [n, d]；q 的第 i 行位于序列位置 q_pos0 + i，只能看到位置 <= 自己的 key。"""
    t, d = q.shape
    n = k.shape[0]
    hd = d // heads
    qh = q.reshape(t, heads, hd).transpose(1, 0, 2)
    kh = k.reshape(n, heads, hd).transpose(1, 0, 2)
    vh = v.reshape(n, heads, hd).transpose(1, 0, 2)
    s = qh @ kh.transpose(0, 2, 1) / np.sqrt(hd)
    mask = np.arange(n)[None, :] > (q_pos0 + np.arange(t))[:, None]
    s = np.where(mask, -np.inf, s)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    return (s @ vh).transpose(1, 0, 2).reshape(t, d)


def forward(w, ids, cache=None):
    """ids 是本次要算的 token；cache 为 None 时整段从头算，否则接在 cache 里已有的位置后面。返回最后一个位置的 logits。"""
    x = w['emb'][ids]
    pos0 = 0 if cache is None else cache['len']
    for i, L in enumerate(w['layers']):
        q, k, v = np.split(rms(x) @ L['qkv'], 3, axis=-1)
        if cache is not None:
            k = cache['k'][i] = np.concatenate([cache['k'][i], k])
            v = cache['v'][i] = np.concatenate([cache['v'][i], v])
        x = x + attend(q, k, v, w['heads'], pos0) @ L['o']
        x = x + np.maximum(rms(x) @ L['up'], 0) @ L['down']
    if cache is not None:
        cache['len'] += len(ids)
    return rms(x[-1]) @ w['out']


def generate(w, prompt, n, use_cache):
    ids = list(prompt)
    if use_cache:
        d = w['emb'].shape[1]
        cache = {'len': 0, 'k': [np.zeros((0, d))] * len(w['layers']), 'v': [np.zeros((0, d))] * len(w['layers'])}
        logits = forward(w, ids, cache)
        for _ in range(n):
            ids.append(int(logits.argmax()))
            if len(ids) - len(prompt) == n:
                break
            logits = forward(w, ids[-1:], cache)
    else:
        for _ in range(n):
            ids.append(int(forward(w, ids).argmax()))
    return ids[len(prompt):]


def flops(prompt, gen, layers, d, d_ff=None, vocab=0, params=None, use_cache=True):
    """生成 gen 个 token 的总 FLOP。params 给定时，投影和 MLP 按 2 x params 每 token 计（logits 已含在参数里）。"""
    per_tok = 2 * params if params else 2 * layers * (4 * d * d + 2 * d * d_ff)
    logits = 0 if params else 2 * d * vocab

    def pass_(n_new, n_ctx):
        # n_new 个新位置，最后一个看到 n_ctx 个 key，第一个看到 n_ctx - n_new + 1 个
        seen = n_new * (2 * n_ctx - n_new + 1) / 2
        return n_new * per_tok + 2 * layers * 2 * d * seen + logits

    if use_cache:
        return pass_(prompt, prompt) + sum(pass_(1, prompt + i + 1) for i in range(gen - 1))
    return sum(pass_(prompt + i, prompt + i) for i in range(gen))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--vocab', type=int, default=512)
    ap.add_argument('--prompt', type=int, default=32)
    ap.add_argument('--gen', type=int, default=96)
    ap.add_argument('--repeat', type=int, default=3, help='每种做法跑几遍，取最短时间')
    ap.add_argument('--params-b', type=float, help='只算账：真实模型参数量（十亿），配合 --layers / --d-model')
    a = ap.parse_args(argv)

    if a.params_b:
        p = a.params_b * 1e9
        on = flops(a.prompt, a.gen, a.layers, a.d_model, params=p, use_cache=True)
        off = flops(a.prompt, a.gen, a.layers, a.d_model, params=p, use_cache=False)
        print(f'模型 {a.params_b:g}B，{a.layers} 层，d={a.d_model}，prompt {a.prompt}，生成 {a.gen}')
        print(f'  带 cache   {on / 1e12:10.2f} TFLOP')
        print(f'  不带 cache {off / 1e12:10.2f} TFLOP')
        print(f'  不带 / 带 = {off / on:.1f}x（只算运算量；带 cache 的 decode 实际受带宽限制，见本章账本）')
        return

    d_ff = 4 * a.d_model
    w = init(a.layers, a.d_model, a.heads, d_ff, a.vocab)
    prompt = list(np.random.default_rng(1).integers(0, a.vocab, a.prompt))
    res = {}
    for use_cache in (True, False):
        best, toks = None, None
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            toks = generate(w, prompt, a.gen, use_cache)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        res[use_cache] = (best, toks, flops(a.prompt, a.gen, a.layers, a.d_model, d_ff, a.vocab, use_cache=use_cache))

    print(f'模型：{a.layers} 层，d={a.d_model}，{a.heads} 头，d_ff={d_ff}，vocab={a.vocab}，float64，随机权重')
    print(f'prompt {a.prompt} 个 token，greedy 生成 {a.gen} 个，每种做法跑 {a.repeat} 遍取最短')
    print(f'{"":12}{"时间 (s)":>9}{"GFLOP":>10}{"每 token (ms)":>15}')
    for use_cache, name in ((True, '带 cache    '), (False, '不带 cache  ')):
        t, _, f = res[use_cache]
        print(f'{name}{t:10.3f}{f / 1e9:10.2f}{t / a.gen * 1e3:16.2f}')
    same = res[True][1] == res[False][1]
    print(f'不带 / 带：时间 {res[False][0] / res[True][0]:.1f}x，运算量 {res[False][2] / res[True][2]:.1f}x')
    print(f'两种做法生成的 token {"完全相同" if same else "不同（不应发生）"}')
    if not same:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
