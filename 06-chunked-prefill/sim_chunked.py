"""Chunked prefill 与异步调度的时间线模拟（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   v1/core/sched/scheduler.py:795-1004  running 循环在前，decode 先拿预算
         v1/core/sched/scheduler.py:1306-1344 waiting 循环：开 chunked 时按剩余预算截断；关掉时整条放不下就 break
         v1/core/sched/scheduler.py:754-769   long_prefill_token_threshold
         v1/engine/core.py:715-800            step_with_batch_queue：异步调度时 batch 队列深度 2，CPU 调度与 GPU 前向重叠
  sglang srt/managers/scheduler.py:4053       get_next_batch_to_run：有新 prefill 就先跑 prefill batch，否则跑 decode
         srt/managers/schedule_policy.py:1150 add_chunked_req：同一时刻最多一条请求在切块
         srt/managers/scheduler.py:4441-4460  --enable-mixed-chunk：prefill batch 里顺带 decode（mix_with_running）
         srt/managers/scheduler.py:2064       event_loop_overlap：默认开启的 CPU/GPU 重叠

四种调度模式：
  vllm          V1 默认：decode 先排，剩余预算切 prompt（stall-free，Sarathi-Serve 的做法）
  vllm-nochunk  V1 关掉 chunked prefill：decode 先排，整条 prompt 放得进剩余预算才收
  sglang        SGLang 默认：prefill batch 与 decode batch 交替，prefill 优先
  sglang-mixed  SGLang --enable-mixed-chunk：prefill batch 顺带所有 running 的 decode

时间模型（H100 SXM + Qwen3-8B BF16）：
  GPU = max(读权重 16.4 GB + 读 KV 144 KiB/token, 线性层 2·P·n + 注意力 FLOPs) 按 3.35 TB/s、989 TFLOPS × 60% × n/(n+192)，再加 0.3 ms
  CPU = 0.8 ms + 8 µs × batch 中请求数（调度 + 准备输入 + 处理输出）
  同步：一步耗时 = CPU + GPU；异步 / overlap：一步耗时 = max(CPU, GPU)
只用来比较相对快慢，绝对值别拿去报数。

用法：
  python3 sim_chunked.py                         # 默认 vllm，预算 8192，混合负载 6 req/s
  python3 sim_chunked.py --sweep                 # 预算 512 … 16384 扫一遍，四种模式对照
  python3 sim_chunked.py --scenario long         # 64 条 decode + 一条 32k prompt：看单步尖刺
  python3 sim_chunked.py --scenario long --sweep
  python3 sim_chunked.py --threshold 2048
  python3 sim_chunked.py --sync                  # 关掉异步调度 / overlap
  python3 sim_chunked.py --scenario burst --sweep # 2000 条同时到达：预算对吞吐的影响
  python3 sim_chunked.py --scenario cpu          # 纯 decode，batch 从 1 到 512：CPU 占比与异步收益
"""
import argparse
import random

W_BYTES, KV_BYTES, HBM, FLOPS, PARAMS = 16.4e9, 147456, 3.35e12, 989e12 * 0.6, 8.2e9
ATTN_FLOPS_PER_PAIR = 4 * 36 * 32 * 128      # QK^T + PV，每个 (q, k) 对、36 层、32 头 × 128
GPU_FIXED_MS = 0.3
GEMM_HALF = 192               # 算力利用率随一步 token 数 n 爬升：eff = n / (n + 192)
CPU_BASE_MS, CPU_PER_REQ_MS = 0.8, 0.008
KV_CAP = 24957 * 16
NEW_TOKEN_RATIO = 0.7          # SGLANG_INIT_NEW_TOKEN_RATIO 默认值（简化成常数，不做衰减）


def gpu_ms(batch):
    """batch: [(chunk 长度 c, 本段起点 s)]。"""
    n = sum(c for c, _ in batch)
    kv = sum(s + c for c, s in batch)
    mem = (W_BYTES + kv * KV_BYTES) / HBM
    flops = 2 * PARAMS * n + sum(c * (s + c / 2) * ATTN_FLOPS_PER_PAIR for c, s in batch)
    eff = n / (n + GEMM_HALF)                        # token 少时 GEMM 吃不满：512 → 73%，8192 → 98%
    return max(mem, flops / (FLOPS * eff)) * 1e3 + GPU_FIXED_MS


def cpu_ms(nreq):
    return CPU_BASE_MS + CPU_PER_REQ_MS * nreq


class Req:
    __slots__ = ("id", "prompt", "out", "arrival", "computed", "gen", "emits", "admitted")

    def __init__(self, rid, prompt, out, arrival):
        self.id, self.prompt, self.out, self.arrival = rid, prompt, out, arrival
        self.computed = 0     # num_computed_tokens
        self.gen = 0          # 已生成 token 数
        self.emits = []       # 每个输出 token 交到用户手里的时刻（ms）
        self.admitted = False

    @property
    def prefilling(self):
        return self.computed < self.prompt

    def need(self):
        # num_tokens − num_computed_tokens；decode 时是 1
        return self.prompt + self.gen - self.computed if self.prefilling else 1


# --------------------------------------------------------------------------- 调度器
class Vllm:
    def __init__(self, a, chunked=True):
        self.a, self.chunked = a, chunked
        self.budget = a.budget if chunked else max(a.budget, a.max_model_len)
        self.running, self.waiting = [], []

    def live_kv(self):
        return sum(r.prompt + r.gen for r in self.running)

    def schedule(self):
        a, budget, out = self.a, self.budget, []
        eligible = len(self.running) + len(self.waiting)
        thr = a.threshold if eligible > 1 else 0
        for r in self.running:                      # 795：running 先排
            if budget <= 0:
                break
            n = r.need()
            if thr and n > thr:
                n = thr
            n = min(n, budget)
            if n == 0:
                continue
            out.append((r, n))
            budget -= n
        while self.waiting and budget > 0 and len(self.running) < a.max_num_seqs:
            r = self.waiting[0]
            if self.live_kv() + r.prompt > KV_CAP:   # full_sequence_must_fit
                break
            n = r.prompt
            if thr and n > thr:
                n = thr
            if n > budget:
                if not self.chunked:                 # 1337-1344：关掉 chunked 时放不下整条就 break
                    break
                n = budget
            self.waiting.pop(0)
            self.running.append(r)
            out.append((r, n))
            budget -= n
        return out

    def finish(self, r):
        self.running.remove(r)


class Sglang:
    def __init__(self, a, mixed=False):
        self.a, self.mixed = a, mixed
        self.running, self.waiting, self.chunked_req = [], [], None

    def live_kv(self):
        return sum(r.computed for r in self.running)

    def rem_total(self):
        # schedule_policy.py:617 PrefillAdder：空闲 KV 先扣掉 running 的预留 min(max_new − 已生成, 4096) × new_token_ratio
        reserve = sum(min(r.out - r.gen, 4096) * NEW_TOKEN_RATIO + (r.prompt - r.computed) for r in self.running)
        return KV_CAP - sum(r.computed for r in self.running) - reserve

    def schedule(self):
        a = self.a
        rem = a.budget                               # chunked_prefill_size
        rem_input = max(a.max_prefill_tokens, a.budget)
        rem_total = self.rem_total()
        decodes = [r for r in self.running if not r.prefilling and r is not self.chunked_req]
        if self.mixed and decodes:
            rem -= len(decodes)                      # num_mixed_decode_tokens
        out = []
        if self.chunked_req is not None:             # 先续上正在切块的那条
            r = self.chunked_req
            n = min(r.need(), rem)
            out.append((r, n))
            rem -= n
            rem_input -= n
            if n == r.need():
                self.chunked_req = None
        while self.chunked_req is None and self.waiting and rem > 0 and rem_input > 0 \
                and len(self.running) < a.max_num_seqs:
            r = self.waiting[0]
            if r.prompt + min(r.out, 4096) * NEW_TOKEN_RATIO > rem_total:   # 891：NO_TOKEN，本步不再收
                break
            rem_total -= r.prompt + min(r.out, 4096) * NEW_TOKEN_RATIO
            self.waiting.pop(0)
            self.running.append(r)
            if r.prompt > rem:                       # 切块：本步只算 rem 个，其余下一步续
                out.append((r, rem))
                self.chunked_req = r
                rem = 0
            else:
                out.append((r, r.prompt))
                rem -= r.prompt
                rem_input -= r.prompt
        if out:                                      # 4053：有 prefill 就先跑 prefill
            if self.mixed and decodes:
                out += [(r, 1) for r in decodes]
            return out
        return [(r, 1) for r in decodes]

    def finish(self, r):
        self.running.remove(r)


# --------------------------------------------------------------------------- 负载
def workload(a):
    rng = random.Random(a.seed)
    reqs = []
    if a.scenario == "long":
        for i in range(64):
            reqs.append(Req(i, rng.randint(200, 1000), 400, 0.0))
        reqs.append(Req(64, 32768, 32, a.long_at))
        return reqs
    if a.scenario == "cpu":
        return [Req(i, 128, 256, 0.0) for i in range(a.bs)]
    if a.scenario == "burst":                        # 离线：全部请求同时到达，看吞吐
        for i in range(a.n):
            if rng.random() < 0.15:
                reqs.append(Req(i, rng.randint(6000, 12000), rng.randint(64, 256), 0.0))
            else:
                reqs.append(Req(i, rng.randint(200, 1000), rng.randint(128, 512), 0.0))
        return reqs
    t = 0.0
    i = 0
    while t < a.duration * 1e3:
        t += rng.expovariate(a.rate) * 1e3
        if rng.random() < 0.15:
            reqs.append(Req(i, rng.randint(6000, 12000), rng.randint(64, 256), t))
        else:
            reqs.append(Req(i, rng.randint(200, 1000), rng.randint(128, 512), t))
        i += 1
    return reqs


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run(a, mode, verbose=False):
    reqs = workload(a)
    S = {"vllm": lambda: Vllm(a), "vllm-nochunk": lambda: Vllm(a, chunked=False),
         "sglang": lambda: Sglang(a), "sglang-mixed": lambda: Sglang(a, mixed=True)}[mode]()
    pending = sorted(reqs, key=lambda r: r.arrival)
    clock, steps, gpu_busy, cpu_total, peak_kv, max_step = 0.0, 0, 0.0, 0.0, 0, 0.0
    max_tokens_step = 0
    trace = []
    while pending or S.running or S.waiting:
        while pending and pending[0].arrival <= clock:
            S.waiting.append(pending.pop(0))
        batch = S.schedule()
        if not batch:
            clock = pending[0].arrival
            continue
        steps += 1
        g = gpu_ms([(n, r.computed) for r, n in batch])
        c = cpu_ms(len(batch))
        wall = c + g if a.sync else max(c, g)
        clock += wall
        gpu_busy += g
        cpu_total += c
        max_step = max(max_step, wall)
        ntok = sum(n for _, n in batch)
        max_tokens_step = max(max_tokens_step, ntok)
        emit_at = clock if a.sync else clock + 0.4 * c
        done = []
        for r, n in batch:
            r.computed += n
            if not r.prefilling:                     # 这一段算到了末尾，采样出一个 token
                r.gen += 1
                r.emits.append(emit_at)
                if r.gen >= r.out:
                    done.append(r)
        for r in done:
            S.finish(r)
        if hasattr(S, "live_kv"):
            peak_kv = max(peak_kv, S.live_kv())
        if verbose:
            trace.append((steps, clock, wall, ntok, sum(1 for r, n in batch if n == 1),
                          [(r.id, n) for r, n in batch if n > 1]))
    ttft = [r.emits[0] - r.arrival for r in reqs]
    itl = [b - a_ for r in reqs for a_, b in zip(r.emits, r.emits[1:])]
    out_tok = sum(r.out for r in reqs)
    t0 = min(r.arrival for r in reqs)
    res = dict(mode=mode, steps=steps, makespan=clock - t0, gpu_util=gpu_busy / (clock - t0),
               cpu_frac=cpu_total / (cpu_total + gpu_busy), out_tps=out_tok / ((clock - t0) / 1e3),
               ttft50=pct(ttft, 0.5), ttft99=pct(ttft, 0.99), itl50=pct(itl, 0.5), itl99=pct(itl, 0.99),
               itlmax=max(itl) if itl else 0.0, max_step=max_step, max_tokens_step=max_tokens_step,
               n=len(reqs), peak_kv=peak_kv, trace=trace, reqs=reqs)
    return res


def fmt(x):
    return f"{x / 1e3:.2f} s" if x >= 1000 else f"{x:.1f} ms"


def summary(res):
    print(f"[{res['mode']:<12}] 步数 {res['steps']:>6}  耗时 {res['makespan'] / 1e3:7.1f} s  输出 {res['out_tps']:>6,.0f} tok/s  "
          f"TTFT p50 {fmt(res['ttft50']):>8} p99 {fmt(res['ttft99']):>8}  "
          f"ITL p50 {fmt(res['itl50']):>8} p99 {fmt(res['itl99']):>8} max {fmt(res['itlmax']):>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["mixed", "long", "cpu", "burst"], default="mixed")
    ap.add_argument("--mode", choices=["vllm", "vllm-nochunk", "sglang", "sglang-mixed"], default="vllm")
    ap.add_argument("--budget", type=int, default=8192, help="vLLM max_num_batched_tokens / SGLang chunked_prefill_size")
    ap.add_argument("--threshold", type=int, default=0, help="long_prefill_token_threshold")
    ap.add_argument("--max-num-seqs", dest="max_num_seqs", type=int, default=1024)
    ap.add_argument("--max-model-len", dest="max_model_len", type=int, default=40960)
    ap.add_argument("--max-prefill-tokens", dest="max_prefill_tokens", type=int, default=16384)
    ap.add_argument("--rate", type=float, default=6.0, help="mixed 场景每秒到达请求数（泊松）")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--long-at", dest="long_at", type=float, default=2000.0, help="long 场景 32k prompt 的到达时刻（ms）")
    ap.add_argument("--n", type=int, default=2000, help="burst 场景的请求数")
    ap.add_argument("--bs", type=int, default=256, help="cpu 场景的 batch 大小")
    ap.add_argument("--sync", action="store_true", help="关掉异步调度 / overlap")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.scenario == "cpu":
        print(f"{'batch':>5} {'GPU/步':>8} {'CPU/步':>8} {'同步/步':>8} {'异步/步':>8} {'同步 GPU 利用率':>14} {'异步提速':>8}")
        for bs in (1, 8, 32, 64, 128, 256, 512):
            g = gpu_ms([(1, 300)] * bs)
            c = cpu_ms(bs)
            print(f"{bs:>5} {g:>7.2f}ms {c:>7.2f}ms {g + c:>7.2f}ms {max(g, c):>7.2f}ms {g / (g + c):>13.0%} {(g + c) / max(g, c):>7.2f}x")
        return

    budgets = [512, 1024, 2048, 4096, 8192, 16384]
    modes = ["vllm", "vllm-nochunk", "sglang", "sglang-mixed"]
    desc = {"mixed": f"混合负载：{a.rate:g} req/s 泊松到达 {a.duration:g} s，85% 短 prompt 200–1000 / 输出 128–512，15% 长 prompt 6k–12k / 输出 64–256",
            "burst": f"离线：{a.n} 条请求同时到达，长短混合比例同 mixed",
            "long": f"64 条短请求在 decode，第 {a.long_at / 1e3:g} s 来一条 32,768 token 的 prompt"}[a.scenario]
    print(f"== {desc}；{'同步' if a.sync else '异步/overlap'}")
    if a.sweep:
        print(f"{'预算':>6} {'模式':<13} {'步数':>6} {'输出 tok/s':>10} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} {'ITL p99':>8} {'ITL max':>8}")
        for b in budgets:
            a.budget = b
            for m in modes:
                if m == "vllm-nochunk" and b != budgets[-1]:
                    continue
                r = run(a, m)
                lab = "—" if m == "vllm-nochunk" else str(b)
                print(f"{lab:>6} {m:<13} {r['steps']:>6} {r['out_tps']:>10,.0f} {fmt(r['ttft50']):>9} {fmt(r['ttft99']):>9} "
                      f"{fmt(r['itl50']):>8} {fmt(r['itl99']):>8} {fmt(r['itlmax']):>8}")
        return
    r = run(a, a.mode, verbose=True)
    if a.scenario == "long":
        print(f"{'step':>5} {'时刻':>9} {'本步耗时':>9} {'token':>6} {'decode':>6}  prefill 段")
        tr = r["trace"]
        hit = [k for k, t in enumerate(tr) if any(rid == 64 for rid, _ in t[5])]
        show = set()
        if hit:
            show = {hit[0] - 1, *hit, hit[-1] + 1}
        for k, (st, clk, wall, ntok, nd, pf) in enumerate(tr):
            if k in show and 0 <= k < len(tr):
                print(f"{st:>5} {clk / 1e3:>8.3f}s {wall:>7.1f}ms {ntok:>6} {nd:>6}  {' '.join(f'R{i}:{n}' for i, n in pf)}")
        lr = r["reqs"][64]
        print(f"32k prompt 的 TTFT {fmt(lr.emits[0] - lr.arrival)}；短请求 ITL max {fmt(r['itlmax'])}")
    summary(r)
    if r["peak_kv"]:
        over = "（超出容量：本模拟不建模抢占，真实系统这里会抢占/retract，见 VLLM-05）" if r["peak_kv"] > KV_CAP else ""
        print(f"         峰值 KV {r['peak_kv']:,} / {KV_CAP:,} token{over}，单步最多 {r['max_tokens_step']:,} token、最长 {fmt(r['max_step'])}")


if __name__ == "__main__":
    main()
