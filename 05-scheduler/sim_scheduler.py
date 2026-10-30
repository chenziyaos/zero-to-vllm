"""vLLM V1 调度器的逐步复刻（只依赖标准库），附一个 SGLang 调度的简化对照。

vLLM 部分逐条对照主干 0d6f92a 的 vllm/v1/core/sched/scheduler.py：
  - 没有 prefill/decode 阶段，只有 num_computed_tokens 追赶 num_tokens       701-710
  - token_budget = max_num_batched_tokens                                    719-722
  - long_prefill_token_threshold 只在 >1 个请求时生效                         757-769
  - 先 running 后 waiting；running 里 num_new_tokens == 0 用 continue          795-1036
  - allocate_slots 失败：FCFS 抢 running[-1]，PRIORITY 抢 max(priority, arrival)  913-980
  - 抢占即重算：释放块、num_computed_tokens = 0、放回 waiting 队首             1799-1843
  - 本步发生过抢占就不收新请求                                                  1050
  - 新请求 allocate_slots(full_sequence_must_fit=True)                         1435-1452
    （kv_cache_manager.py:530-547：整条 prompt 放不下就不收）
  - 调度完才推进 num_computed_tokens（_update_after_schedule）                1845
  - 异步调度：num_output_placeholders（async_scheduler.py）

SGLang 部分（sgl-project/sglang 8fabf5e）只保留和 vLLM 有差别的三件事：
  - 收新请求前按 min(max_new - 已生成, 4096) × new_token_ratio 给 running 请求预留
    （schedule_policy.py PrefillAdder._get_running_request_total_token_offset）
  - 有新 prefill 就先跑 prefill batch，否则跑 decode（scheduler.py get_next_batch_to_run）
  - decode 时 KV 不够就 retract：按 (len(output), -len(input)) 倒序，从末尾弹，至少留 1 条
    （schedule_batch.py retract_decode / _get_decode_retraction_order），
    然后 new_token_ratio = (已生成 + 20 × n) / (Σ max_new + 1)，平时每步衰减

用法：
    python3 sim_scheduler.py                          # 场景 unified：一条长 prompt + 两条 decode
    python3 sim_scheduler.py --scenario preempt       # KV 只有 64 块，看 FCFS 抢占
    python3 sim_scheduler.py --scenario preempt --policy priority
    python3 sim_scheduler.py --scenario preempt --no-reserve-full-isl
    python3 sim_scheduler.py --scenario preempt --async
    python3 sim_scheduler.py --scenario h100          # 24,957 块、预算 8192、600 条同时到达
    python3 sim_scheduler.py --scenario h100 --engine sglang
    python3 sim_scheduler.py --scenario h100 --compare   # 两个引擎同负载对比
"""
import argparse
import random

BLOCK = 16
CLIP_MAX_NEW_TOKENS = 4096
# 粗略的单步耗时模型（H100 + Qwen3-8B BF16，与 VLLM-02/03 的账本一致）：
#   读权重 16.4 GB 与读 KV（144 KiB/token）按 3.35 TB/s，算力按 989 TFLOPS × 60%，取大者，再加 1.2 ms 固定开销
W_BYTES, KV_BYTES, HBM, FLOPS, PARAMS, FIXED_MS = 16.4e9, 147456, 3.35e12, 989e12 * 0.6, 8.2e9, 1.2


def step_ms(n_tokens, kv_tokens):
    mem = (W_BYTES + kv_tokens * KV_BYTES) / HBM
    comp = 2 * PARAMS * n_tokens / FLOPS
    return max(mem, comp) * 1e3 + FIXED_MS


def cdiv(a, b):
    return -(-a // b)


class Req:
    def __init__(self, rid, prompt, max_tokens, real_out, arrival=0, priority=0):
        self.id = rid
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.real_out = min(real_out, max_tokens)  # 第 real_out 个 token 是 EOS
        self.arrival = arrival
        self.priority = priority
        self.out = 0
        self.computed = 0
        self.placeholders = 0
        self.stale = False
        self.blocks = 0          # vLLM：持有的块数；SGLang：持有的 token 槽位数
        self.status = "WAITING"
        self.preemptions = 0
        self.first_token_step = None
        self.finish_step = None

    @property
    def num_tokens(self):
        return self.prompt + self.out

    def __repr__(self):
        return f"R{self.id}"


# --------------------------------------------------------------------------- vLLM
class VllmScheduler:
    def __init__(self, a):
        self.a = a
        self.free = a.num_blocks
        self.watermark = int(a.watermark * a.num_blocks)
        self.running, self.waiting = [], []
        self.preempt_total = 0
        self.recomputed = 0
        self.step = 0
        self.clock = 0.0

    # kv_cache_manager.allocate_slots 的块数部分
    def allocate(self, r, n_new, full_must_fit=False, has_scheduled=False):
        wm = self.watermark if has_scheduled and r.status in ("WAITING", "PREEMPTED") else 0
        if full_must_fit:
            need_full = cdiv(min(r.num_tokens, self.a.max_model_len), BLOCK) - r.blocks
            if need_full + wm > self.free:
                return False
        need = cdiv(min(r.computed + n_new + self.a.lookahead, self.a.max_model_len), BLOCK) - r.blocks
        if need + wm > self.free:
            return False
        need = max(need, 0)
        self.free -= need
        r.blocks += need
        return True

    def release(self, r):
        self.free += r.blocks
        r.blocks = 0

    def preempt(self, r, ev):
        ev.append(f"抢占 {r}(丢 {r.computed} tok 的 KV)")
        self.release(r)
        self.recomputed += r.computed
        r.status = "PREEMPTED"
        r.computed = 0
        r.stale = r.placeholders > 0
        r.placeholders = 0
        r.preemptions += 1
        self.preempt_total += 1
        self.waiting.insert(0, r)  # FCFS: appendleft；PRIORITY 是堆，顺序由 key 决定
        if self.a.policy == "priority":
            self.waiting.sort(key=lambda x: (x.priority, x.arrival))

    def schedule(self):
        a = self.a
        budget = a.budget
        sched = {}
        ev = []
        preempted = []
        n_eligible = len(self.running) + len(self.waiting)
        thr = a.threshold if n_eligible > 1 else 0
        # ---- running
        i = 0
        while i < len(self.running) and budget > 0:
            r = self.running[i]
            if r.placeholders > 0 and r.computed + 2 - r.placeholders >= r.prompt + r.max_tokens:
                i += 1
                continue
            n = r.num_tokens + r.placeholders - r.computed
            if 0 < thr < n:
                n = thr
            n = min(n, budget, a.max_model_len - r.computed - 1)
            if n <= 0:
                i += 1
                continue
            ok = False
            while True:
                if self.allocate(r, n):
                    ok = True
                    break
                if a.policy == "priority":
                    victim = max(self.running, key=lambda x: (x.priority, x.arrival))
                    vi = self.running.index(victim)
                    del self.running[vi]
                    if vi < i:
                        i -= 1
                    if victim.id in sched:
                        budget += sched.pop(victim.id)
                else:
                    victim = self.running.pop()
                self.preempt(victim, ev)
                preempted.append(victim)
                if victim is r:
                    break
            if not ok:
                if a.policy == "priority" and r.status == "PREEMPTED":
                    continue
                break
            sched[r.id] = n
            budget -= n
            i += 1
        # ---- waiting：本步有抢占就不收新请求
        if not preempted:
            while budget > 0 and self.waiting and len(self.running) < a.max_num_seqs:
                r = self.waiting[0]
                n = r.num_tokens - r.computed
                if 0 < thr < n:
                    n = thr
                if not a.chunked and n > budget:
                    break
                n = min(n, budget)
                if not self.allocate(r, n, full_must_fit=a.reserve_full_isl, has_scheduled=bool(self.running)):
                    break
                self.waiting.pop(0)
                if r.status == "PREEMPTED":
                    ev.append(f"恢复 {r}")
                r.status = "RUNNING"
                self.running.append(r)
                sched[r.id] = n
                budget -= n
        # ---- _update_after_schedule：调度完就推进 computed
        by_id = {r.id: r for r in self.running}
        for rid, n in sched.items():
            r = by_id[rid]
            r.computed += n
            if a.async_sched and r.computed >= r.num_tokens + r.placeholders:
                r.placeholders += 1
        return sched, ev

    def update(self, sched):
        """update_from_output：给算完整条序列的请求追加 1 个 token，判停。"""
        done = []
        by_id = {r.id: r for r in self.running + self.waiting}
        for rid in sched:
            r = by_id.get(rid)
            if r is None or r.status == "FINISHED":
                continue
            if self.a.async_sched:
                if r.status == "PREEMPTED":
                    # 被抢占前已发出的那一步：输出照样交付（stale output），只是不再动计数
                    if r.stale:
                        r.stale = False
                        r.out += 1
                    continue
                if r.placeholders == 0:
                    continue
                r.placeholders -= 1
            elif r.status != "RUNNING":
                continue
            elif r.computed < r.num_tokens:
                continue  # prefill 只算了一段，采样结果丢弃
            r.out += 1
            if r.first_token_step is None:
                r.first_token_step = self.step
                r.ttft_ms = self.clock
            if r.out >= r.real_out:
                done.append(r)
        for r in done:
            r.status = "FINISHED"
            r.finish_step = self.step
            r.e2e_ms = self.clock
            self.running.remove(r)
            self.release(r)
        return done

    def used_pct(self):
        return 100 * (1 - self.free / self.a.num_blocks)


# --------------------------------------------------------------------------- SGLang
class SglangScheduler:
    """token 粒度（page_size=1）的简化版，只为对比预留与 retract 的行为。"""

    def __init__(self, a):
        self.a = a
        self.cap = a.num_blocks * BLOCK
        self.free = self.cap
        self.running, self.waiting = [], []
        self.init_ratio = min(0.7 * a.conservativeness, 1.0)
        self.min_ratio = self.init_ratio * 0.14
        self.decay = (self.init_ratio - self.min_ratio) / 600
        self.ratio = self.init_ratio
        self.preempt_total = 0
        self.recomputed = 0
        self.step = 0
        self.clock = 0.0
        self.chunked = None

    def offset(self, r):
        return min(r.max_tokens - r.out, CLIP_MAX_NEW_TOKENS) * self.ratio

    def schedule(self):
        a = self.a
        ev = []
        # ---- 先尝试 prefill batch
        rem_total = self.free - sum(self.offset(r) for r in self.running)
        rem_input = a.max_prefill_tokens
        rem_chunk = a.budget  # chunked_prefill_size
        sched = {}
        queue = ([self.chunked] if self.chunked else []) + [r for r in self.waiting if r is not self.chunked]
        new_chunked = None
        for r in queue:
            if len(self.running) + len(sched) >= a.max_num_seqs:
                break
            extend = r.num_tokens - r.computed
            max_new = min(max(r.max_tokens - r.out, 0), CLIP_MAX_NEW_TOKENS)
            if r is not self.chunked and extend + max_new > rem_total:
                ev.append("NO_TOKEN（预留不够，新请求等待）")
                break
            if rem_input <= 0 or rem_chunk <= 0:
                break
            n = min(extend, rem_chunk)
            if n < extend:
                new_chunked = r
            sched[r.id] = n
            rem_total -= n + (max_new if r is not self.chunked else 0)
            rem_input -= n
            rem_chunk -= n
            if new_chunked:
                break
        if sched:
            for r in list(queue):
                if r.id not in sched:
                    continue
                n = sched[r.id]
                self.free -= n
                r.blocks += n
                r.computed += n
                if r in self.waiting:
                    self.waiting.remove(r)
                if r.status == "PREEMPTED":
                    ev.append(f"恢复 {r}")
                r.status = "PREFILL"
            self.chunked = new_chunked
            self.prefill_batch = [r for r in queue if r.id in sched]
            return sched, ev
        # ---- decode batch
        self.prefill_batch = []
        need = len(self.running)
        if need > self.free and len(self.running) > 1:
            order = sorted(self.running, key=lambda r: (r.out, -r.prompt), reverse=True)
            retracted = []
            while need > self.free and len(order) > 1:
                r = order.pop()
                self.running.remove(r)
                self.free += r.blocks
                self.recomputed += r.computed
                r.blocks = 0
                r.computed = 0
                r.status = "PREEMPTED"
                r.preemptions += 1
                retracted.append(r)
                need -= 1
            self.preempt_total += len(retracted)
            old = self.ratio
            self.ratio = min(1.0, (sum(r.out for r in self.running) + 20 * len(self.running))
                             / (sum(r.max_tokens for r in self.running) + 1))
            self.waiting.extend(retracted)
            ev.append(f"retract {retracted}，new_token_ratio {old:.3f}→{self.ratio:.3f}")
        else:
            self.ratio = max(self.ratio - self.decay, self.min_ratio)
        for r in self.running:
            sched[r.id] = 1
            self.free -= 1
            r.blocks += 1
            r.computed += 1
        return sched, ev

    def update(self, sched):
        done = []
        for r in self.prefill_batch:
            if r is self.chunked:
                continue
            r.status = "RUNNING"
            self.running.append(r)
            r.out += 1
            if r.first_token_step is None:
                r.first_token_step = self.step
                r.ttft_ms = self.clock
            if r.out >= r.real_out:
                done.append(r)
        if not self.prefill_batch:
            for r in self.running:
                if r.id in sched:
                    r.out += 1
                    if r.out >= r.real_out and r not in done:
                        done.append(r)
        for r in done:
            r.status = "FINISHED"
            r.finish_step = self.step
            r.e2e_ms = self.clock
            if r in self.running:
                self.running.remove(r)
            self.free += r.blocks
            r.blocks = 0
        return done

    def used_pct(self):
        return 100 * (1 - self.free / self.cap)


# --------------------------------------------------------------------------- 场景
def workload(a):
    if a.scenario == "unified":
        # R0、R1 已经在 decode，R2 是一条 5000 token 的长 prompt
        return [Req(0, 300, 40, 40), Req(1, 200, 40, 40), Req(2, 5000, 20, 20, arrival=1)]
    if a.scenario == "preempt":
        # 8 条 prompt 100、各生成 200 个 token；KV 只有 64 块 = 1024 token
        return [Req(i, 100, 200, 200, arrival=i, priority=(i % 2)) for i in range(8)]
    rng = random.Random(a.seed)
    reqs = []
    for i in range(a.n):
        p = rng.randint(1000, 3000)
        o = rng.randint(200, 1500)
        reqs.append(Req(i, p, a.max_tokens, o, arrival=i))
    return reqs


def apply_scenario_defaults(a):
    d = {"unified": dict(num_blocks=4096, budget=2048, max_num_seqs=128),
         "preempt": dict(num_blocks=64, budget=2048, max_num_seqs=128),
         "h100": dict(num_blocks=24957, budget=8192, max_num_seqs=1024)}[a.scenario]
    for k, v in d.items():
        if getattr(a, k) is None:
            setattr(a, k, v)


def run(a, engine, verbose):
    reqs = workload(a)
    S = VllmScheduler(a) if engine == "vllm" else SglangScheduler(a)
    pending = [r for r in reqs if r.arrival > 0]
    S.waiting = [r for r in reqs if r.arrival == 0] if a.scenario != "h100" else list(reqs)
    if a.scenario == "h100":
        pending = []
    if a.scenario == "unified":
        # 让 R0、R1 先跑完 prefill，从 decode 状态开始观察
        pass
    tokens_total = 0
    clock = 0.0
    no_token_steps = 0
    timeline = []
    inflight = None
    if verbose:
        print(f"{'step':>4}  {'本步调度（请求:token 数）':<44} {'run':>3} {'wait':>4} {'KV%':>5}  事件")
    while True:
        S.step += 1
        for r in [r for r in pending if r.arrival <= S.step - 1]:
            S.waiting.append(r)
            pending.remove(r)
        sched, ev = S.schedule()
        tokens_total += sum(sched.values())
        kv_live = sum(r.computed for r in S.running)
        clock += step_ms(sum(sched.values()), kv_live)
        S.clock = clock
        timeline.append((clock, len(S.running), len(S.waiting), S.used_pct(), S.preempt_total))
        if engine == "vllm" and a.async_sched:
            done = S.update(inflight) if inflight else []
            inflight = sched
        else:
            done = S.update(sched)
        if done:
            ev.append("完成 " + ",".join(str(r) for r in done))
        real_ev = [e for e in ev if not e.startswith("NO_TOKEN")]
        if any(e.startswith("NO_TOKEN") for e in ev):
            no_token_steps += 1
            if not real_ev and no_token_steps > 1:
                ev = []
        if verbose and (S.step <= a.show or real_ev or ev) and S.step <= a.max_show:
            s = " ".join(f"R{k}:{v}" for k, v in sched.items())
            if len(s) > 44:
                s = s[:41] + "..."
            print(f"{S.step:>4}  {s:<44} {len(S.running):>3} {len(S.waiting):>4} {S.used_pct():>5.1f}  {'；'.join(ev)}")
        if all(r.status == "FINISHED" for r in reqs) and not pending:
            break
        if S.step > 200000:
            print("超过 200000 步，停止")
            break
    useful = sum(r.prompt + r.real_out for r in reqs)
    ttft = sorted(r.ttft_ms for r in reqs)
    res = dict(steps=S.step, preempt=S.preempt_total, recomputed=S.recomputed, tokens=tokens_total,
               useful=useful, ttft_p50=ttft[len(ttft) // 2], ttft_p99=ttft[int(len(ttft) * 0.99) - 1],
               maxpre=max(r.preemptions for r in reqs), total_s=clock / 1e3,
               out_tps=sum(r.real_out for r in reqs) / (clock / 1e3), timeline=timeline)
    return res


def fmt_ms(x):
    return f"{x / 1e3:.1f} s" if x >= 10000 else f"{x:.0f} ms"


def summary(name, res):
    print(f"[{name}] 步数 {res['steps']}，抢占/retract {res['preempt']} 次（单条最多 {res['maxpre']} 次），"
          f"重算 {res['recomputed']:,} token，前向总 token {res['tokens']:,}（有效 {res['useful']:,}）\n"
          f"         模型耗时 {res['total_s']:.1f} s，输出 {res['out_tps']:,.0f} tok/s，"
          f"TTFT p50 {fmt_ms(res['ttft_p50'])} / p99 {fmt_ms(res['ttft_p99'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["unified", "preempt", "h100"], default="unified")
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--compare", action="store_true", help="同负载跑 vLLM 和 SGLang 两种调度")
    ap.add_argument("--num-blocks", dest="num_blocks", type=int)
    ap.add_argument("--budget", type=int, help="max_num_batched_tokens / SGLang chunked_prefill_size")
    ap.add_argument("--max-num-seqs", dest="max_num_seqs", type=int)
    ap.add_argument("--max-model-len", dest="max_model_len", type=int, default=40960)
    ap.add_argument("--threshold", type=int, default=0, help="long_prefill_token_threshold")
    ap.add_argument("--policy", choices=["fcfs", "priority"], default="fcfs")
    ap.add_argument("--no-reserve-full-isl", dest="reserve_full_isl", action="store_false")
    ap.add_argument("--no-chunked", dest="chunked", action="store_false")
    ap.add_argument("--watermark", type=float, default=0.0)
    ap.add_argument("--async", dest="async_sched", action="store_true", help="异步调度：输出晚一步回来")
    ap.add_argument("--lookahead", type=int, default=0)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prefill-tokens", dest="max_prefill_tokens", type=int, default=16384)
    ap.add_argument("--conservativeness", type=float, default=1.0)
    ap.add_argument("--show", type=int, default=12, help="前 N 步无条件打印，之后只打印有事件的步")
    ap.add_argument("--max-show", dest="max_show", type=int, default=60)
    a = ap.parse_args()
    apply_scenario_defaults(a)
    print(f"== 场景 {a.scenario}：KV {a.num_blocks} 块 × {BLOCK} = {a.num_blocks * BLOCK:,} token，"
          f"预算 {a.budget}，max_num_seqs {a.max_num_seqs}，policy {a.policy}，"
          f"reserve_full_isl {a.reserve_full_isl}，async {a.async_sched}")
    if a.compare:
        r1 = run(a, "vllm", False)
        r2 = run(a, "sglang", False)
        summary("vLLM ", r1)
        summary("SGLang", r2)
        return
    res = run(a, a.engine, a.scenario != "h100")
    summary("vLLM" if a.engine == "vllm" else "SGLang", res)


if __name__ == "__main__":
    main()
