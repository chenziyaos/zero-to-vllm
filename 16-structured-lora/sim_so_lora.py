"""结构化输出与多 LoRA 模拟：bitmask 账本、CPU 填 mask 与 GPU 前向的重叠、语法编译与超时、投机解码下的
逐 token 推进与回滚（vLLM StructuredOutputManager 逐行移植）、auto 后端选择、jump-forward、LoRA 预分配显存、
Punica 三种算子、多 LoRA 调度（vLLM max_loras vs SGLang max_loras_per_batch + drainer）、前缀缓存隔离、
SGLang 对照（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   config/structured_outputs.py:23               backend 默认 "auto"
         sampling_params.py:39, 44                     JSON 嵌套上限 64，structural tag 再加 16
         sampling_params.py:217–260                    _exceeds_json_nesting 按层广度优先数嵌套
         sampling_params.py:1377–1444                  auto：Tekken+Lark → guidance；否则先试 xgrammar，失败再
                                                       outlines / guidance（本文件 auto_backend 移植）
         structured_output/backend_xgrammar.py:202     max_rollback_tokens = num_speculative_tokens
         structured_output/backend_xgrammar.py:264–310 accept_tokens / validate_tokens / rollback
         structured_output/backend_xgrammar.py:362–476 has_xgrammar_unsupported_json_features（逐行移植）
         structured_output/backend_guidance.py:93–116  has_guidance_unsupported_json_features：任意一层有 patternProperties
         structured_output/utils.py:48, 62             strip_speculative_padding；compile_regex_with_timeout
         envs.py:232, 1717                             VLLM_REGEX_COMPILATION_TIMEOUT_S=5，VLLM_XGRAMMAR_CACHE_MB=512
         structured_output/__init__.py:46–47           _CONSTRAINED_FROM_START=(0,0)，_NO_SPEC_TOKENS=(1,1)
         structured_output/__init__.py:81–96           并行填 mask 阈值 128、每批 16、线程 max(1,min(cpu//2,8))；
                                                       编译线程 max(1,(cpu+1)//2)
         structured_output/__init__.py:143–150         一个引擎只能用一个后端
         structured_output/__init__.py:231–540         _fill_bitmasks / _get_constraint_bounds / trim_spec_tokens /
                                                       grammar_bitmask / accept_tokens（本文件 Manager 移植）
         v1/engine/core.py:685–693                     execute_model(non_block) 之后才算 bitmask，再 sample_tokens
         v1/worker/gpu/structured_outputs.py:12–171    块大小 8192；grammar_invalid_drafts；bit 为 0 的 logit 置 -inf
         v1/core/sched/scheduler.py:1038–1102          scheduled_loras = running 里所有 LoRA（含本步没排上的），满了就跳过
         config/lora.py:32–67                          max_lora_rank=16，max_loras=1，fully_sharded_loras=False
         lora/layers/base_linear.py:104–155            A/B 按 max_loras × max_lora_rank 预分配
         v1/core/kv_cache_utils.py:589–607             前缀哈希的 extra key = ("lora", name, path)
  sglang srt/constrained/xgrammar_backend.py:63        MAX_ROLLBACK_TOKENS=200；try_jump_forward 在调度路径上没有调用
         srt/managers/grammar_manager.py:255–300       轮询次数 ≥ SGLANG_GRAMMAR_MAX_POLL_ITERATIONS(10000) 判失败
         srt/environ.py:410–411                        轮询间隔 0.005 s
         srt/server_args（lora 命名空间）              max_loras_per_batch=8（含 base），lora_backend=csgmv，
                                                       lora_eviction_policy=lru，lora_drain_wait_threshold=0.0
         srt/managers/scheduler.py:4251–4262, 4481–4507 running_loras；can_schedule_lora_req
         srt/lora/lora_manager.py:427–456              validate_lora_batch（本文件 sgl_validate 移植）
         srt/lora/lora_drainer.py                      DRAIN_SCHEDULE_TOLERANCE=1.2（本文件 Drainer 移植）
         srt/managers/schedule_batch.py:1170–1173      前缀 extra_key 拼上 lora_id
         srt/lora/lora_registry.py:37–54               lora_id = uuid4；--lora-paths 用 uuid5(name\\0path)

假设：
  - 语法用玩具实现：有限 JSON 语言（几条完整字符串，前缀可接受即可接受）和两条正则（[0-9]+、z[0-9]+）。token 用
    字符串表示，EOS 只在语法已完整时被接受。只用来复现 vLLM 测试 test_structured_output_manager.py 的行模式，
    不代表 xgrammar 的 PDA 实现。
  - overlap 场景的 GPU 单步时间按 XGrammar 论文 Table 2（MLC-LLM、Llama-3.1-8B）bs1 6.2 ms、bs16 9.0 ms 线性外推；
    每行 mask 的 CPU 时间取论文 Table 3 各优化档位的逐 token 延迟。H2D 有效带宽假设 25 GB/s（pinned）。
  - sched 场景每步固定 20 ms，不计 LoRA 个数对算子的影响（Punica 的 SGMV 在 bs64 Distinct 下也只到 116 µs/层）。
  - punica 场景的曲线是对论文 Fig 8 / Fig 9 端点的线性插值，只有端点是实测。

用法：
  python3 sim_so_lora.py --scenario mask       # bitmask 有多大：vocab × 行数 × 投机步数，每步搬多少字节
  python3 sim_so_lora.py --scenario overlap    # CPU 填 mask 串行 vs 与 GPU 前向重叠，各优化档位的代价
  python3 sim_so_lora.py --scenario compile    # 编译线程、缓存命中、vLLM 正则 5 s 超时 vs SGLang 轮询上限
  python3 sim_so_lora.py --scenario spec       # 投机解码：逐 token 推进、回滚、trim、行模式（复现官方测试）
  python3 sim_so_lora.py --scenario auto       # auto 后端选择：哪些 schema 掉到 guidance / outlines / 报错
  python3 sim_so_lora.py --scenario jump       # jump-forward 的收益与两家都没接上的现状
  python3 sim_so_lora.py --scenario loramem    # LoRA A/B 预分配：rank × max_loras × TP × fully_sharded
  python3 sim_so_lora.py --scenario punica     # Loop / Gather-BMM / SGMV 的 I/O 与延迟
  python3 sim_so_lora.py --scenario sched      # Zipf 负载下 max_loras=1/8、SGLang 8 槽、drainer 的排队时间（约 5 秒）
  python3 sim_so_lora.py --scenario prefix     # 换权重、换名字、多机重启时前缀缓存会不会串
  python3 sim_so_lora.py --scenario sglang     # 两家对照表
"""
import argparse
import hashlib
import math
import os
import random
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from specs import MODELS  # noqa: E402

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30

# ---- vLLM 常量
FILL_PARALLEL_THRESHOLD = 128        # structured_output/__init__.py:81
FILL_PARALLEL_BATCH = 16             # :83
BITMASK_BLOCK = 8192                 # worker/gpu/structured_outputs.py:12
JSON_NESTING = 64                    # sampling_params.py:39
STRUCT_TAG_NESTING = 16              # sampling_params.py:44
REGEX_TIMEOUT_S = 5                  # envs.py:232
XGR_CACHE_MB = 512                   # envs.py:1717
# ---- SGLang 常量
SGL_POLL_S = 0.005                   # environ.py:410–411
SGL_MAX_POLL = 10000                 # grammar_manager.py
SGL_ROLLBACK = 200                   # xgrammar_backend.py:63
SGL_MAX_LORAS_PER_BATCH = 8          # 含 base 请求
DRAIN_TOL = 1.2                      # lora_drainer.py

H2D_GBS = 25                         # 假设
XGR_TABLE3 = [('PDA 基线', 65.776), ('+ 节点合并', 38.280), ('+ 自适应 token mask 缓存', 0.154),
              ('+ 规则内联', 0.035), ('+ 上下文扩展', 0.018)]   # XGrammar 论文 Table 3，ms/token


def fill_workers(cpu):
    return max(1, min(cpu // 2, 8))


def compile_workers(cpu):
    return max(1, (cpu + 1) // 2)


def mask_cols(vocab):
    return math.ceil(vocab / 32)


def mask_bytes(rows, vocab):
    return rows * mask_cols(vocab) * 4


# --------------------------------------------------------------------------- 1. bitmask 账本
def scenario_mask():
    print('== 每行 bitmask = ceil(vocab/32) 个 int32；vLLM 预分配 max_num_seqs × (1 + k) 行')
    for key in ('qwen3-8b', 'llama3.1-8b'):
        m = MODELS[key]
        print(f'  {m.name:13s} vocab {m.vocab:,} → {mask_cols(m.vocab):,} 列，每行 {mask_cols(m.vocab) * 4:,} B')
    print()
    print(f'  {"模型":13s} {"max_num_seqs":>12s} {"k":>2s} {"预分配行":>8s} {"预分配":>9s}')
    for key in ('qwen3-8b', 'llama3.1-8b'):
        m = MODELS[key]
        for seqs in (256, 1024):
            for k in (0, 3, 5):
                rows = seqs * (1 + k)
                print(f'  {m.name:13s} {seqs:12d} {k:2d} {rows:8d} {mask_bytes(rows, m.vocab) / MiB:7.1f} MiB')
    print('  max_num_seqs 默认：显存 ≥ 160 GiB，或 ≥ 70 GiB 且不是 A100/GB10 时 1024，否则 256（arg_utils 2979–3011）')
    print()
    print(f'== 每步真正搬的是 结构化请求数 × (k+1) 行（grammar_bitmask 最后切到 num_masks），H2D 按 {H2D_GBS} GB/s 估')
    m = MODELS['qwen3-8b']
    print(f'  {"结构化请求":>10s} {"k":>2s} {"行":>6s} {"字节":>10s} {"H2D":>8s} {"kernel grid":>14s}')
    for n in (1, 32, 128, 256):
        for k in (0, 3):
            rows = n * (1 + k)
            b = mask_bytes(rows, m.vocab)
            grid = (rows, math.ceil(m.vocab / BITMASK_BLOCK))
            print(f'  {n:10d} {k:2d} {rows:6d} {b / MiB:8.2f}MiB {b / (H2D_GBS * 1e9) * 1e3:6.3f}ms {str(grid):>14s}')
    print('  bitmask 先在 scheduler 进程转成 numpy 发给 worker（序列化），再在 copy_stream 上异步 H2D；kernel 前 wait_stream。')
    print('  k=3、256 个结构化请求时每步约 18 MiB，H2D 就要 0.7 ms 以上，投机步数会线性放大这笔账。')


# --------------------------------------------------------------------------- 2. CPU 填 mask 与 GPU 前向的重叠
def gpu_step_ms(bs):
    """XGrammar Table 2：bs1 6.2 ms、bs16 9.0 ms，线性外推（假设）。"""
    return 6.2 + (9.0 - 6.2) / 15 * (bs - 1)


def fill_ms(n_rows, per_row_ms, cpu=16, k=0):
    """vLLM grammar_bitmask：> 128 个请求且 k == 0 才走线程池并行，否则串行。"""
    if n_rows > FILL_PARALLEL_THRESHOLD and k == 0:
        batches = math.ceil(n_rows / FILL_PARALLEL_BATCH)
        waves = math.ceil(batches / fill_workers(cpu))
        return waves * FILL_PARALLEL_BATCH * per_row_ms
    return n_rows * per_row_ms


def scenario_overlap():
    print('== 时间线（engine/core.py 685–693）：execute_model(non_block=True) 先把前向发出去，CPU 再算 bitmask，')
    print('   future.result() 等前向结束，然后 sample_tokens 带着 bitmask 采样。CPU 填 mask 被藏在 GPU 前向里。')
    print()
    print(f'  {"每行 mask 成本":24s} {"bs":>4s} {"GPU 步":>7s} {"填 mask":>9s} {"串行 TPOT":>10s} {"重叠 TPOT":>10s}')
    for name, per in XGR_TABLE3[1:]:
        for bs in (1, 16, 64, 256):
            g = gpu_step_ms(bs)
            f = fill_ms(bs, per)
            print(f'  {name:24s} {bs:4d} {g:6.1f}ms {f:7.2f}ms {g + f:8.1f}ms {max(g, f):8.1f}ms')
    print(f'  PDA 基线 {XGR_TABLE3[0][1]} ms/行，bs1 就比一步前向慢 10 倍，重叠也救不回来。')
    print('  bs256 的填 mask 反而比 bs64 短：过了 128 的并行阈值，按 cpu=16 开 8 个线程。')
    print()
    cpu = os.cpu_count() or 8
    print(f'== 并行阈值：> {FILL_PARALLEL_THRESHOLD} 个结构化请求且没有投机，按 {FILL_PARALLEL_BATCH} 个一批丢给 '
          f'max(1, min(cpu//2, 8)) 个线程（本机 cpu={cpu} → {fill_workers(cpu)}）')
    for c in (4, 16, 64):
        f_ser = 256 * 0.018
        f_par = fill_ms(256, 0.018, cpu=c)
        print(f'  cpu={c:3d} 线程 {fill_workers(c)}：256 行 串行 {f_ser:.2f} ms → 并行 {f_par:.3f} ms（还要加 GIL 和提交开销）')
    print('  有投机时永远串行：每个请求要逐 draft 推进语法再回滚，状态有先后依赖。')
    print('  XGrammar Table 2：MLC-LLM 上开不开 XGrammar，JSON Schema bs1 6.2 vs 6.3 ms，bs16 9.0 vs 9.2 ms。')


# --------------------------------------------------------------------------- 3. 编译与超时
def compile_makespan(n_req, n_distinct, c_ms, workers, cache=True):
    """n_req 个请求、n_distinct 种 schema，每种首次编译 c_ms，命中缓存当 0。返回最后一个就绪的时刻。"""
    jobs = n_distinct if cache else n_req
    return math.ceil(jobs / workers) * c_ms


def sgl_grammar_timeout_s(step_s):
    return SGL_MAX_POLL * max(SGL_POLL_S, step_s)


def scenario_compile():
    print('== vLLM：请求到达即 submit 到编译线程池，状态 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR，编译好才进 WAITING')
    print(f'  编译线程 max(1, (cpu+1)//2)；xgrammar GrammarCompiler max_threads=8、缓存上限 {XGR_CACHE_MB} MB')
    print(f'  {"请求":>5s} {"不同 schema":>11s} {"单次编译":>8s} {"cpu":>4s} {"有缓存":>8s} {"无缓存":>8s}')
    for n, d, c in ((64, 1, 200), (64, 64, 200), (256, 8, 50), (256, 256, 50)):
        for cpu in (8, 32):
            w = compile_workers(cpu)
            print(f'  {n:5d} {d:11d} {c:6d}ms {cpu:4d} {compile_makespan(n, d, c, w):6d}ms '
                  f'{compile_makespan(n, d, c, w, cache=False):6d}ms')
    print('  同一个 schema 的请求排队等同一份编译结果；每请求一个新 schema（工具调用里很常见）才会把线程池打满。')
    print()
    print('== 超时：谁来终止一个编不完的语法')
    print(f'  vLLM：只有正则有 {REGEX_TIMEOUT_S} s 编译超时（compile_regex_with_timeout，校验阶段和编译阶段各一次）；')
    print('        JSON schema / grammar 没有超时，编译抛异常时请求以 "Invalid structured outputs" 结束。')
    print('  SGLang：grammar_manager 按轮询次数计，wait_ct ≥ 10000 判失败，单次间隔 max(0.005 s, 一步 GPU 耗时)')
    for step in (0.002, 0.005, 0.02, 0.05):
        print(f'    一步 {step * 1e3:4.0f} ms → 最长等 {sgl_grammar_timeout_s(step):6.0f} s')
    print('  TP 下 SGLang 每轮 all_gather：就绪取交集、失败取并集，所有 rank 一起放行。')


# --------------------------------------------------------------------------- 4. 投机解码：Manager 逐行移植
EOS = '<eos>'
PAD = -1
THINK_END = '\n'          # 测试里的显式推理结束符
IMPLICIT_END = 'z'        # 测试里的隐式结束符：字符串里可以出现，对象开头不行


class ToyGrammar:
    """玩具语法：is_prefix / is_complete 两个判定函数。接口照 backend_xgrammar.XgrammarGrammar。"""

    def __init__(self, is_prefix, is_complete, text=''):
        self.is_prefix, self.is_complete = is_prefix, is_complete
        self.hist = [text]
        self.term = [False]

    @property
    def text(self):
        return self.hist[-1]

    def is_terminated(self):
        return self.term[-1]

    def _accept_one(self, tok):
        if tok == EOS:
            if not self.is_complete(self.text):
                return False
            self.hist.append(self.text)
            self.term.append(True)
            return True
        new = self.text + tok
        if not self.is_prefix(new):
            return False
        self.hist.append(new)
        self.term.append(False)
        return True

    def accept_tokens(self, tokens):
        """backend_xgrammar.py:264：已终止直接 True；逐个接受，拒绝就回滚已接受的并返回 False；终止即停。"""
        if self.is_terminated():
            return True
        n = 0
        for t in tokens:
            if not self._accept_one(t):
                self.rollback(n)
                return False
            n += 1
            if self.is_terminated():
                break
        return True

    def validate_tokens(self, tokens):
        """backend_xgrammar.py:286：试着接受，记下接受了哪些，再全部回滚，不改变状态。"""
        if self.is_terminated():
            return []
        ok = []
        for t in tokens:
            if not self._accept_one(t):
                break
            ok.append(t)
            if self.is_terminated():
                break
        self.rollback(len(ok))
        return ok

    def rollback(self, n):
        for _ in range(n):
            self.hist.pop()
            self.term.pop()


JSON_DOCS = ('{"a": "b"}', '{"a": "z"}', '{}', '{ "b": "c"}')


def json_grammar(prefix=''):
    return ToyGrammar(lambda s: any(d.startswith(s) for d in JSON_DOCS), lambda s: s in JSON_DOCS, prefix)


def regex_grammar(lead=''):
    """lead='' 是 [0-9]+，lead='z' 是 z[0-9]+。"""
    def is_prefix(s):
        if len(s) <= len(lead):
            return lead.startswith(s)
        return s.startswith(lead) and s[len(lead):].isdigit()

    def is_complete(s):
        return s.startswith(lead) and len(s) > len(lead) and s[len(lead):].isdigit()
    return ToyGrammar(is_prefix, is_complete)


class MockReasoner:
    """测试里的 MockReasoner：marker 出现在 delta 里就算推理结束；marker 为 None 时永远已结束。"""

    def __init__(self, marker):
        self.marker = marker

    def is_reasoning_end(self, ids):
        return True if self.marker is None else self.marker in list(ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        return self.is_reasoning_end(delta_ids)


class EngineReasoner(MockReasoner):
    """测试里的 MockEngineReasoner：marker 是隐式结束（本身也是内容），end_marker 是显式结束。"""

    def __init__(self, marker, end_marker=None):
        super().__init__(marker)
        self.end_marker = end_marker
        self.end_ids = {marker, end_marker} - {None}
        self.implicit_ids = {marker}

    def is_reasoning_end(self, ids):
        return any(t in self.end_ids for t in ids)

    def find_reasoning_end(self, ids):
        for off, t in enumerate(ids):
            if t in self.end_ids:
                return off, t == self.marker
        return len(ids), False


class Req:
    def __init__(self, grammar, reasoner=None, reasoning_ended=False, prompt=()):
        self.grammar, self.reasoner, self.reasoning_ended = grammar, reasoner, reasoning_ended
        self.prompt = list(prompt)
        self.all_token_ids = list(prompt)
        self.masked_ids = None


def strip_padding(tokens):
    """utils.py:48：在第一个 < 0 的位置截断（这里 token 是字符串，PAD 是 -1）。"""
    out = []
    for t in tokens:
        if t == PAD:
            break
        out.append(t)
    return out


class Manager:
    def __init__(self, k, enable_in_reasoning=False, parallel_threshold=FILL_PARALLEL_THRESHOLD):
        self.k, self.enable_in_reasoning, self.parallel_threshold = k, enable_in_reasoning, parallel_threshold

    def masked_reasoning_ends(self, req):
        """__init__.py:259：只有引擎式 reasoner 才有；语法不能以它开头的隐式结束符，推理期间屏蔽。"""
        if req.masked_ids is None:
            r = req.reasoner
            req.masked_ids = tuple(t for t in sorted(r.implicit_ids) if not req.grammar.validate_tokens([t])) \
                if isinstance(r, EngineReasoner) else ()
        return req.masked_ids

    def bounds(self, req, spec, committed=False):
        """__init__.py:284 _get_constraint_bounds → (grammar_start, constraint_start)。"""
        if self.enable_in_reasoning or req.reasoning_ended or req.reasoner is None:
            return 0, 0
        r = req.reasoner
        if req.reasoning_ended is None:
            if r.is_reasoning_end(req.prompt):
                return 0, 0
            req.reasoning_ended = False
        n = len(spec)
        if n <= 0:
            return 1, 1
        if isinstance(r, EngineReasoner) and r.end_ids:
            off, implicit = r.find_reasoning_end(spec)
            cs = off + 1
            return (off, cs) if implicit else (cs, cs)
        if committed:
            if not spec or not r.is_reasoning_end_streaming(req.all_token_ids, spec):
                return n + 1, n + 1
            inp, delta = list(req.all_token_ids), list(spec)
        else:
            inp = list(req.all_token_ids) + list(spec)
            if not r.is_reasoning_end_streaming(inp, spec):
                return n + 1, n + 1
            delta = list(spec)
        for i in range(n - 1, 0, -1):
            inp.pop()
            delta.pop()
            if not r.is_reasoning_end_streaming(inp, delta):
                return i + 1, i + 1
        return 1, 1

    def trim(self, req, spec):
        """__init__.py:363 trim_spec_tokens：语法前缀 + 验证通过的 + 1 个（给第一个不合法的位置留采样机会）。"""
        spec = strip_padding(spec)
        gs, _ = self.bounds(req, spec)
        if gs >= len(spec):
            return spec
        if req.grammar.is_terminated():
            return spec[:gs]
        n_ok = len(req.grammar.validate_tokens(spec[gs:]))
        return spec[:gs] + spec[gs:gs + n_ok + 1]

    @staticmethod
    def _row(grammar, apply, masked):
        """_fill_bitmasks：C=语法行，R=全放开但屏蔽 masked_ids，U=全放开。"""
        if apply and not grammar.is_terminated():
            return 'C'
        return 'R' if masked else 'U'

    def bitmask(self, reqs, spec_by_req):
        """__init__.py:388 grammar_bitmask。返回每个请求 mask_stride = k+1 行的模式串。"""
        if len(reqs) > self.parallel_threshold and self.k == 0:
            out = []
            for req in reqs:
                apply = self.bounds(req, ())[1] == 0
                out.append(self._row(req.grammar, apply, () if apply else self.masked_reasoning_ends(req)))
            return out
        out = []
        for idx, req in enumerate(reqs):
            g = req.grammar
            tokens = spec_by_req[idx]
            gs, cs = self.bounds(req, strip_padding(tokens))
            masked = self.masked_reasoning_ends(req) if cs else ()
            rows, adv, invalid = [], 0, False
            for i, tok in enumerate(list(tokens) + [PAD]):
                if invalid:
                    rows.append(self._row(g, False, ()))
                    continue
                rows.append(self._row(g, i >= cs, masked if i < cs else ()))
                if tok == PAD:
                    invalid = True
                elif i >= gs:
                    if not g.is_terminated() and g.accept_tokens([tok]):
                        adv += 1
                    else:
                        invalid = True
            if adv:
                g.rollback(adv)
            out.append(''.join(rows))
        return out

    def accept(self, req, new_tokens):
        """__init__.py:520 accept_tokens：只把推理结束之后的 token 喂给语法，并锁存 reasoning_ended。"""
        if not new_tokens:
            return True
        gs, _ = self.bounds(req, new_tokens, committed=True)
        if gs > len(new_tokens):
            return True
        req.reasoning_ended = True
        return req.grammar.accept_tokens(new_tokens[gs:])


def flow(grammar, drafts, reasoner=None, reasoning_ended=False):
    m = Manager(k=8)
    req = Req(grammar, reasoner, reasoning_ended)
    return m.bitmask([req], [list(drafts)])[0][:len(drafts) + 1]


def grammar_invalid_drafts(num_drafts, num_acceptable):
    """worker/gpu/structured_outputs.py:15：draft i 在本地位置 i+1，位置 > 可接受数的 logit 行作废。"""
    return [pos > num_acceptable for pos in range(num_drafts + 1)]


TE, Z = THINK_END, IMPLICIT_END
FLOW_CASES = [
    # (名字, 语法, drafts, reasoner, reasoning_ended, 期望行模式)
    ('inactive', lambda: json_grammar(), (' ', Z, ' '), MockReasoner(TE), False, 'UUUU'),
    ('active_all_valid', lambda: json_grammar('{"a"'), (':', ' "', 'b', '"}', PAD, PAD), MockReasoner(TE), True, 'CCCCCUU'),
    ('active_first_token_invalid', lambda: json_grammar('{"a"'), (Z, ':', ' "', 'b'), MockReasoner(TE), True, 'CUUUU'),
    ('active_later_token_invalid', lambda: json_grammar('{"a"'), (':', Z, ' "', 'b'), MockReasoner(TE), True, 'CCUUU'),
    ('active_first_token_terminates', lambda: json_grammar('{"a": "b"}'), (EOS, ' ', ' '), MockReasoner(TE), True, 'CUUU'),
    ('active_later_token_terminates', lambda: json_grammar('{"a"'), (':', ' "', 'b', '"}', EOS, ' '),
     MockReasoner(TE), True, 'CCCCCUU'),
    ('becomes_active_on_first_token', lambda: json_grammar(), (TE, '{', ' "', 'b'), MockReasoner(TE), False, 'UCCCC'),
    ('becomes_active_on_middle_token', lambda: json_grammar(), (' ', TE, '{', ' "'), MockReasoner(TE), False, 'UUCCC'),
    ('becomes_active_on_last_token', lambda: json_grammar(), (' ', ' ', TE), MockReasoner(TE), False, 'UUUC'),
    ('becomes_active_first_token_invalid', lambda: json_grammar(), (' ', TE, Z, '{'), MockReasoner(TE), False, 'UUCUU'),
    ('becomes_active_later_token_invalid', lambda: json_grammar(), (' ', TE, '{', Z), MockReasoner(TE), False, 'UUCCU'),
    ('becomes_active_terminates', lambda: json_grammar(), (' ', TE, '{', '}', EOS, ' '), MockReasoner(TE), False, 'UUCCCUU'),
    ('implicit_end_accepted', lambda: json_grammar('{"a": "'), (Z,), EngineReasoner(Z), False, 'UC'),
    ('implicit_end_rejected', lambda: json_grammar(), (' ', Z, '{', '}'), EngineReasoner(Z), False, 'RRUUU'),
]
TRIM_CASES = [((TE, '1', '2', PAD), 3), ((TE, '1', Z, '2'), 3), (('1', Z, '2', PAD), 3), ((Z, TE, '1', Z, '2', PAD), 4)]
REGEX_CASES = [(('1', '2'), 'CCC'), (('1', EOS, '2'), 'CCUU'), ((EOS, '1'), 'CUU')]


def show(t):
    return {TE: 'TE', EOS: 'EOS', PAD: '-1'}.get(t, repr(t))


def scenario_spec():
    print('== FLOW_CASES（tests/v1/structured_output/test_structured_output_manager.py），k=8')
    print('   行 i 是"接受了前 i 个 draft 之后"的 mask：C=语法约束，U=全放开，R=放开但屏蔽隐式结束符')
    ok_all = True
    for name, g, drafts, r, ended, want in FLOW_CASES:
        got = flow(g(), drafts, r, ended)
        ok_all &= got == want
        print(f'  {name:36s} {" ".join(show(t) for t in drafts):30s} → {got:8s} {"✓" if got == want else "✗ 期望 " + want}')
    print()
    print('== trim_spec_tokens：正则 [0-9]+，推理以 TE 结束')
    for drafts, want in TRIM_CASES:
        m = Manager(k=8)
        req = Req(regex_grammar(), MockReasoner(TE), False)
        got = len(m.trim(req, list(drafts)))
        ok_all &= got == want
        print(f'  {" ".join(show(t) for t in drafts):24s} → 留 {got} 个 {"✓" if got == want else "✗"}')
    print()
    print('== 推理行屏蔽隐式结束符（引擎式 reasoner，隐式 z + 显式 TE）')
    for name, g, drafts, want in (('json 不能以 z 开头', json_grammar(), (' ', ' '), 'RRR'),
                                  ('regex z[0-9]+ 能以 z 开头', regex_grammar('z'), (' ', Z, '1'), 'UUCC')):
        got = flow(g, drafts, EngineReasoner(Z, TE))
        ok_all &= got == want
        print(f'  {name:26s} {" ".join(show(t) for t in drafts):12s} → {got} {"✓" if got == want else "✗"}')
    m = Manager(k=0, parallel_threshold=0)
    req = Req(json_grammar(), EngineReasoner(Z, TE), False)
    before = m.bitmask([req], [[]])[0]
    m.accept(req, [TE])
    after = m.bitmask([req], [[]])[0]
    ok_all &= (before, after) == ('R', 'C')
    print(f'  并行填充路径（k=0）：推理中 {before}，接受 TE 后 {after}')
    print()
    print('== 正则 [0-9]+ 的终止（推理已结束）')
    for drafts, want in REGEX_CASES:
        got = flow(regex_grammar(), drafts, None, True)
        ok_all &= got == want
        print(f'  {" ".join(show(t) for t in drafts):12s} → {got} {"✓" if got == want else "✗"}')
    print(f'  全部与官方测试一致：{ok_all}')
    print()
    print('== GPU 侧 grammar_invalid_drafts：draft 4 个，语法只认前 2 个 → 位置 3、4 的 logit 行作废（拒绝采样必然拒）')
    print(f'  {grammar_invalid_drafts(4, 2)}')
    print()
    print('== 代价：投机时一律串行，每请求 k+1 行，逐 draft accept 一次、最后 rollback 一次')
    for k in (0, 1, 3, 5):
        print(f'  k={k}：每请求 {k + 1} 行 mask、最多 {k} 次 accept；xgrammar max_rollback_tokens = {k}'
              f'（SGLang 固定 {SGL_ROLLBACK}）')


# --------------------------------------------------------------------------- 5. auto 后端选择
STRING_FORMATS = {'email', 'date', 'time', 'date-time', 'duration', 'ipv4', 'ipv6', 'hostname', 'uuid', 'uri',
                  'uri-reference', 'uri-template', 'json-pointer', 'relative-json-pointer'}   # backend_xgrammar.py:326


def _types(o):
    t = o.get('type')
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    return set()


def _pat_len(o):
    return ('pattern' in o or 'format' in o) and ('minLength' in o or 'maxLength' in o)


def xgr_unsupported(schema):
    """backend_xgrammar.py:362–476 逐行移植。返回命中的原因，没有命中返回 None。"""
    def chk(o):
        if not isinstance(o, dict):
            return None
        t = _types(o)
        if t & {'integer', 'number'} and 'multipleOf' in o:
            return 'number + multipleOf'
        if 'array' in t and any(k in o for k in ('uniqueItems', 'contains', 'minContains', 'maxContains')):
            return 'array + uniqueItems/contains'
        if 'string' in t and 'format' in o and o['format'] not in STRING_FORMATS:
            return f'string format={o["format"]}'
        if 'string' in t and _pat_len(o):
            return 'string pattern/format + min/maxLength'
        if 'object' in t and 'propertyNames' in o:
            pn = o.get('propertyNames')
            if isinstance(pn, dict) and _pat_len(pn):
                return 'propertyNames 带长度'
            if 'patternProperties' in o:
                return 'propertyNames + patternProperties'
            if 'properties' in o:
                return 'propertyNames + properties'
            if o.get('unevaluatedProperties', True) is not True:
                return 'propertyNames + unevaluatedProperties'
        if 'object' in t and isinstance(o.get('patternProperties'), dict):
            if 'properties' in o:
                return 'patternProperties + properties'
            if len(o['patternProperties']) > 1:
                return '多个 patternProperties'
        if isinstance(o.get('allOf'), list) and len(o['allOf']) >= 2:
            return 'allOf 多分支'
        for v in o.values():
            if isinstance(v, dict):
                r = chk(v)
                if r:
                    return r
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        r = chk(it)
                        if r:
                            return r
        return None
    return chk(schema)


def guidance_unsupported(schema):
    """backend_guidance.py:93：任意一层出现 patternProperties。"""
    if not isinstance(schema, dict):
        return False
    if 'patternProperties' in schema:
        return True
    for v in schema.values():
        if isinstance(v, dict) and guidance_unsupported(v):
            return True
        if isinstance(v, list) and any(isinstance(i, dict) and guidance_unsupported(i) for i in v):
            return True
    return False


def json_depth_exceeds(spec, limit=JSON_NESTING):
    """sampling_params.py:217：广度优先，每层只保留 dict/list 子节点。"""
    level = [spec]
    for _ in range(limit):
        level = [c for n in level if isinstance(n, (dict, list))
                 for c in (n.values() if isinstance(n, dict) else n) if isinstance(c, (dict, list))]
        if not level:
            return False
    return True


def auto_backend(schema, tokenizer='fast', tekken_lark=False):
    """sampling_params.py:1377–1444。tokenizer: fast / slow / mistral / tekken。"""
    if json_depth_exceeds(schema):
        return 'reject', f'嵌套超过 {JSON_NESTING} 层'
    if tekken_lark and tokenizer == 'tekken':
        return 'guidance', 'Tekken + Lark 语法'
    why = xgr_unsupported(schema)
    if why is None:
        return 'xgrammar', '-'
    skip_guidance = tokenizer == 'mistral' or guidance_unsupported(schema)
    if skip_guidance:
        return 'outlines', why + ('；非 Tekken Mistral' if tokenizer == 'mistral' else '；有 patternProperties')
    if tokenizer == 'slow':
        return 'reject', why + '；guidance 不支持慢 tokenizer'
    return 'guidance', why


def nest(n):
    s = {'type': 'string'}
    for _ in range(n):
        s = {'type': 'array', 'items': s}
    return s


AUTO_CASES = [
    ('普通对象', {'type': 'object', 'properties': {'name': {'type': 'string'}}}, 'fast'),
    ('price: multipleOf 0.01', {'type': 'number', 'multipleOf': 0.01}, 'fast'),
    ('tags: uniqueItems', {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True}, 'fast'),
    ('email 格式', {'type': 'string', 'format': 'email'}, 'fast'),
    ('自定义 format', {'type': 'string', 'format': 'phone'}, 'fast'),
    ('pattern + maxLength', {'type': 'string', 'pattern': '^[A-Z]+$', 'maxLength': 8}, 'fast'),
    ('单个 patternProperties', {'type': 'object', 'patternProperties': {'^x_': {'type': 'string'}}}, 'fast'),
    ('patternProperties + properties', {'type': 'object', 'properties': {'id': {'type': 'integer'}},
                                        'patternProperties': {'^x_': {'type': 'string'}}}, 'fast'),
    ('allOf 两个分支', {'allOf': [{'type': 'string'}, {'enum': ['yes', 'no']}]}, 'fast'),
    ('multipleOf + 慢 tokenizer', {'type': 'integer', 'multipleOf': 5}, 'slow'),
    ('multipleOf + Mistral(非 Tekken)', {'type': 'integer', 'multipleOf': 5}, 'mistral'),
    ('嵌套 64 层', nest(63), 'fast'),
    ('嵌套 65 层', nest(64), 'fast'),
]


def scenario_auto():
    print('== backend="auto" 每个请求单独判定（sampling_params 1377–1444）')
    print(f'  {"schema":34s} {"tokenizer":9s} {"后端":9s} 原因')
    for name, sch, tok in AUTO_CASES:
        be, why = auto_backend(sch, tok)
        print(f'  {name:34s} {tok:9s} {be:9s} {why}')
    print()
    print('== 一个引擎只能有一个后端（structured_output/__init__.py 143–150）')
    seq = [AUTO_CASES[0][1], AUTO_CASES[1][1], AUTO_CASES[7][1]]
    engine = None
    for i, sch in enumerate(seq):
        be, _ = auto_backend(sch)
        if engine is None:
            engine = be
            print(f'  第 {i + 1} 个请求 → {be}，引擎锁定为 {be}')
        elif be != engine:
            print(f'  第 {i + 1} 个请求 → {be}，与 {engine} 不同：报 "only supports one backend per engine"')
        else:
            print(f'  第 {i + 1} 个请求 → {be}')
    print('  反过来，如果第一个请求恰好是 multipleOf，引擎会锁成 guidance，之后所有普通 schema 都走 guidance。')
    print('  生产上显式写 --structured-outputs-config.backend，别让第一条请求决定整台实例的后端。')
    print()
    print('  SGLang：--grammar-backend 启动时选定（默认 xgrammar），没有逐请求的 auto 回退。')


# --------------------------------------------------------------------------- 6. jump-forward
def scenario_jump():
    print('== jump-forward：语法唯一确定的后续字符串（如 JSON 的 "key": ）不用一个个采样，直接追加')
    for name, before, after in (('SGLang + Outlines', 44.2, 31.5), ('SGLang + XGrammar', 6.8, 5.4)):
        print(f'  {name:18s} TPOT {before:5.1f} → {after:5.1f} ms（-{(1 - after / before) * 100:.0f}%）')
    print('  来源：XGrammar 论文 Fig 11，RTX 4090 + Ryzen 7950X，Llama-3.1-8B，bs1，JSON Schema 任务')
    print()
    print('== 两家主干的现状')
    print('  vLLM：backend_xgrammar.py 253 行只有指向 find_jump_forward_string 的注释，guidance 后端写着 TODO。')
    print('  SGLang：XGrammarGrammar.try_jump_forward 仍在（xgrammar_backend.py 168），但 scheduler / model_runner /')
    print('          schedule_batch / grammar_manager 里没有调用点。')
    print('  所以两家的结构化输出提速都靠 mask 生成快 + 与 GPU 重叠，不靠跳字。想要跳字的收益，投机解码是现成的替代：')
    print('  语法确定的 token 被 draft 猜中后，一次验证就吃下多个位置。')


# --------------------------------------------------------------------------- 7. LoRA 预分配显存
INTERMEDIATE = {'qwen3-8b': 12288, 'llama3.1-8b': 14336}


def lora_layer_bytes(m, inter, rank, tp=1, fully_sharded=False, dtype=2):
    """base_linear.py:104–155。返回一层 qkv/o/gate_up/down 的 A、B 预分配字节（单卡、单个 LoRA 槽）。"""
    h, q, kv = m.hidden, m.heads * m.head_dim, m.kv_heads * m.head_dim
    col_a = rank // tp if fully_sharded else rank
    # 列并行：A 是 [a_out, 全量输入]，B 是 [每卡输出, rank]；合并层每个 slice 一份
    qkv = sum(col_a * h + (o // tp) * rank for o in (q, kv, kv))
    gate_up = 2 * (col_a * h + (inter // tp) * rank)
    # 行并行：A 是 [rank, 每卡输入]，B 是 [全量输出（fully_sharded 时 /tp）, rank]
    row_b = h // tp if fully_sharded else h
    o_proj = rank * (q // tp) + row_b * rank
    down = rank * (inter // tp) + row_b * rank
    return (qkv + gate_up + o_proj + down) * dtype


def lora_total_bytes(key, rank, max_loras, tp=1, fully_sharded=False):
    m = MODELS[key]
    return lora_layer_bytes(m, INTERMEDIATE[key], rank, tp, fully_sharded) * m.layers * max_loras


def scenario_loramem():
    print('== vLLM 按 max_loras × max_lora_rank 预分配 A/B（base_linear.py 104–155），与实际加载的 adapter rank 无关')
    print('   统计 qkv / o / gate_up / down 四类线性层，bf16，单卡')
    print(f'  {"模型":13s} {"rank":>4s} {"max_loras":>9s} {"TP":>3s} {"每卡":>9s} {"fully_sharded 每卡":>17s}')
    for key in ('qwen3-8b', 'llama3.1-8b'):
        for rank in (16, 64, 256):
            for ml in (1, 8):
                for tp in (1, 4):
                    a = lora_total_bytes(key, rank, ml, tp)
                    b = lora_total_bytes(key, rank, ml, tp, True)
                    print(f'  {MODELS[key].name:13s} {rank:4d} {ml:9d} {tp:3d} {a / MiB:7.0f}MiB {b / MiB:15.0f}MiB')
    print('  max_lora_rank 取的是最大的那个 adapter：混着一个 rank 256 的，8 个槽全按 256 留。')
    print('  fully_sharded 只切列并行的 A 和行并行的 B，TP 越大省得越多，代价是多一次通信。')
    print()
    print('== 一个 adapter 从 CPU 换上 GPU 要搬多少（按实际 rank，H2D 假设 25 GB/s）')
    for key in ('qwen3-8b', 'llama3.1-8b'):
        for rank in (16, 64):
            b = lora_total_bytes(key, rank, 1)
            print(f'  {MODELS[key].name:13s} rank {rank:3d}: {b / MiB:6.0f} MiB → {b / (H2D_GBS * 1e9) * 1e3:5.1f} ms')
    print('  Punica 的量级：PCIe Gen4 x16 上一层约 50 µs、整个模型约 2 ms（Llama-2-7B，rank 16）。')
    print('  SGLang 默认 max_loras_per_batch=8；vLLM 默认 max_loras=1，max_cpu_loras 不设时等于 max_loras。')


# --------------------------------------------------------------------------- 8. Punica 三种算子
SGMV_FIG8 = {'Distinct': (37, 116), 'Uniform': (37, 46), 'Skewed': (37, 46), 'Identical': (37, 40)}
SGMV_FIG9_DISTINCT = {8: (42, 72), 16: (42, 75), 32: (42, 89), 64: (42, 118)}


def interp(ends, bs, bs_max=64):
    lo, hi = ends
    return lo + (hi - lo) * (bs - 1) / (bs_max - 1)


def distinct_models(workload, n):
    if workload == 'Distinct':
        return n
    if workload == 'Identical':
        return 1
    if workload == 'Uniform':
        return math.ceil(math.sqrt(n))
    # Skewed：Punica 定义第 i 个模型的请求数是第 i+1 个的 α 倍（α=1.5）
    w = [1.5 ** -i for i in range(n)]
    cnt, s = [], sum(w)
    for x in w:
        cnt.append(max(0, round(n * x / s)))
    return max(1, sum(1 for c in cnt if c > 0))


def gather_bmm_extra_bytes(n, hi, ho, dtype=2):
    """Gather 读 n·hi·ho 写 sn·hi·ho，BMM 再读一遍 sn·hi·ho：比 SGMV 多 sn·hi·ho·2 个元素。"""
    return n * hi * ho * 2 * dtype


def scenario_punica():
    print('== Punica 的三种实现（论文 §7 算子微基准，A100 80GB）')
    print('  Loop：每个 LoRA 一轮 bs=1 的小 GEMM；Gather-BMM：先把每个请求要的权重拷成一份大张量再 bmm；')
    print('  SGMV：按 segment 直接索引各自的权重，一次 kernel')
    print()
    print(f'  {"workload":10s} {"bs":>3s} {"不同模型":>8s} {"SGMV(µs)":>9s} {"Loop 下限(µs)":>13s} {"Gather 多搬":>11s}')
    for wl in ('Distinct', 'Uniform', 'Skewed', 'Identical'):
        for bs in (1, 16, 64):
            d = distinct_models(wl, bs)
            sg = interp(SGMV_FIG8[wl], bs)
            extra = gather_bmm_extra_bytes(bs, 4096, 16)
            print(f'  {wl:10s} {bs:3d} {d:8d} {sg:9.0f} {d * 37:13d} {extra / KiB:9.0f}KiB')
    print('  SGMV 端点是论文实测（Distinct 37→116 µs，Uniform/Skewed 37→46，Identical 37→40），中间点线性插值。')
    print('  Loop 下限按"每个不同模型一轮 37 µs"估，是假设；Gather 多搬的字节按 h=4096、rank 16 的 A 矩阵算。')
    print()
    print('== rank 的影响（Fig 9，Distinct）：bs1 约 42 µs，bs64')
    for r, ends in SGMV_FIG9_DISTINCT.items():
        print(f'  r={r:3d}: {ends[1]} µs')
    print('  有权重共享时（Uniform/Skewed/Identical）bs 1→64 都在 42–45 µs。')
    print('  层延迟：seq 512 时 bs 1→32 只涨 72%；LoRA 算子约占 1%。Punica 吞吐比基线高 12 倍，每 token 只多 2 ms。')
    print()
    print('== 工业实践：AWS 在 vLLM 上调多 LoRA kernel（vLLM 博客 2026-02-26）')
    print('  Triton 对 LoRA 下标做了值特化，每来一个新 adapter 组合就重编，TTFT 涨 10 倍 → 加 do_not_specialize')
    print('  expand 与 add 融合：144 → 171 OTPS（tuned），TPOT 135 → 124 ms；合计 OTPS +454%、TTFT -87%')
    print('  测试配置 1600 输入 / 600 输出、rank 32、8 个 adapter；他们说常见 rank 是 16–64')


# --------------------------------------------------------------------------- 9. 多 LoRA 调度
class SReq:
    __slots__ = ('rid', 'arr', 'lora', 'max_new', 'out', 'start')

    def __init__(self, rid, arr, lora, max_new):
        self.rid, self.arr, self.lora, self.max_new = rid, arr, lora, max_new
        self.out, self.start = 0, None


def sgl_validate(ids, max_per_batch=SGL_MAX_LORAS_PER_BATCH, pinned=frozenset()):
    """lora_manager.py:427–456。ids 里 None 代表 base 请求，同样占一个槽。"""
    if len(ids) > max_per_batch:
        return False
    if not pinned:
        return True
    in_batch = sum(1 for i in ids if i in pinned)
    return len(ids) - in_batch <= max_per_batch - len(pinned)


class Drainer:
    """lora_drainer.py 移植：等太久的 adapter 让剩余 token 最少的 running adapter 停止进新请求。"""

    def __init__(self, threshold_steps, max_loras):
        self.th, self.max = threshold_steps, max_loras
        self.draining_for = {}            # 被排空的 adapter -> 为谁排空

    @staticmethod
    def _remaining(running, lora):
        return max((r.max_new - r.out for r in running if r.lora == lora), default=0)

    def update(self, waiting, running, now):
        active = {r.lora for r in running}
        for a in [a for a in self.draining_for if a not in active]:
            del self.draining_for[a]
        if len(active) < self.max:
            return
        wait = {}
        for r in waiting:
            if r.lora not in active:
                wait[r.lora] = max(wait.get(r.lora, 0), now - r.arr)
        served = set(self.draining_for.values())
        starving = sorted((a for a, w in wait.items() if w > self.th and a not in served), key=lambda a: -wait[a])
        for s in starving:
            cand = [a for a in active if a not in self.draining_for]
            if not cand:
                break
            victim = min(cand, key=lambda a: self._remaining(running, a))
            self.draining_for[victim] = s

    def can_schedule(self, req, running):
        if req.lora not in self.draining_for:
            return True
        return req.max_new <= self._remaining(running, req.lora) * DRAIN_TOL


def zipf_pick(rng, cdf):
    x = rng.random()
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def gen_workload(seed, steps, rate, n_adapters, base_frac, alpha=1.5):
    rng = random.Random(seed)
    w = [1 / (i + 1) ** alpha for i in range(n_adapters)]
    s, acc, cdf = sum(w), 0.0, []
    for x in w:
        acc += x / s
        cdf.append(acc)
    reqs, rid = [], 0
    for t in range(steps):
        n = 0
        p, L = rng.random(), math.exp(-rate)      # Poisson 采样
        while p > L:
            n += 1
            p *= rng.random()
        for _ in range(n):
            lora = None if rng.random() < base_frac else zipf_pick(rng, cdf) + 1
            reqs.append(SReq(rid, t, lora, rng.randint(32, 512)))
            rid += 1
    return reqs


def run_sched(reqs, steps, policy, max_batch=64, max_loras=8, drain_steps=0):
    """policy: vllm / sglang。每步每个 running 请求出 1 个 token（假设没有 token 预算争用）。"""
    reqs = [SReq(r.rid, r.arr, r.lora, r.max_new) for r in reqs]
    pending, waiting, running, done = list(reversed(reqs)), [], [], []
    drainer = Drainer(drain_steps, max_loras) if (policy == 'sglang' and drain_steps > 0) else None
    for t in range(steps):
        while pending and pending[-1].arr <= t:
            waiting.append(pending.pop())
        if drainer:
            drainer.update(waiting, running, t)
        if policy == 'vllm':
            loras = {r.lora for r in running if r.lora is not None}     # scheduler.py:1038–1047
        else:
            loras = {r.lora for r in running}                           # base 的 None 也算
        keep = []
        for r in waiting:
            if len(running) >= max_batch:
                keep.append(r)
                continue
            if policy == 'vllm':
                ok = r.lora is None or r.lora in loras or len(loras) < max_loras
            else:
                ok = (drainer is None or drainer.can_schedule(r, running)) and \
                     (r.lora in loras or sgl_validate(loras | {r.lora}, max_loras))
            if not ok:
                keep.append(r)
                continue
            r.start = t
            running.append(r)
            if policy != 'vllm' or r.lora is not None:
                loras.add(r.lora)
        waiting = keep
        nxt = []
        for r in running:
            r.out += 1
            (done if r.out >= r.max_new else nxt).append(r)
        running = nxt
    return reqs, waiting


def pct(xs, p):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def sched_stats(reqs, steps, step_ms, by_rank=None):
    cut = int(steps * 0.7)
    ws = [((r.start if r.start is not None else steps) - r.arr) * step_ms / 1e3 for r in reqs if r.arr < cut
          and (by_rank is None or by_rank(r.lora))]
    starved = sum(1 for r in reqs if r.arr < cut and r.start is None and (by_rank is None or by_rank(r.lora)))
    return pct(ws, 50), pct(ws, 99), max(ws) if ws else float('nan'), starved


STEP_MS = 20
SCHED_CONFIGS = [
    ('vLLM max_loras=1（默认）', 'vllm', 1, 0),
    ('vLLM max_loras=4', 'vllm', 4, 0),
    ('vLLM max_loras=8', 'vllm', 8, 0),
    ('SGLang 8 槽（含 base）', 'sglang', 8, 0),
    ('SGLang 8 槽 + drainer 2 s', 'sglang', 8, 100),
]


def scenario_sched():
    steps, rate, n_ad, base = 6000, 0.18, 32, 0.2
    reqs = gen_workload(1, steps, rate, n_ad, base)
    print(f'== 负载：{n_ad} 个 adapter，Zipf α=1.5，{int(base * 100)}% 走 base；到达 {rate / STEP_MS * 1e3:.1f} req/s，')
    print(f'   输出 32–512 token 均匀，每步 {STEP_MS} ms，max_num_seqs=64，模拟 {steps * STEP_MS / 1e3:.0f} s；'
          f'只统计前 70% 时间到达的 {sum(1 for r in reqs if r.arr < steps * 0.7)} 个请求')
    print(f'  {"配置":28s} {"排队 p50":>8s} {"p99":>7s} {"最长":>7s} {"没排上":>6s} | '
          f'{"冷门 adapter p99":>15s} {"base p99":>8s}')
    for name, pol, ml, dr in SCHED_CONFIGS:
        out, _ = run_sched(reqs, steps, pol, max_loras=ml, drain_steps=dr)
        p50, p99, mx, st = sched_stats(out, steps, STEP_MS)
        _, cold99, _, _ = sched_stats(out, steps, STEP_MS, lambda l: l is not None and l > 8)
        _, b99, _, _ = sched_stats(out, steps, STEP_MS, lambda l: l is None)
        print(f'  {name:28s} {p50:7.2f}s {p99:6.1f}s {mx:6.1f}s {st:6d} | {cold99:14.1f}s {b99:7.1f}s')
    print('  冷门 adapter = Zipf 排名 9 以后的那些。')
    print('  vLLM：只要还有一个该 adapter 的请求在 running（哪怕本步因 token 预算没排上），它就占着一个槽；')
    print('        槽满时新 adapter 的请求被 skip 留在 waiting，base 请求不占槽。')
    print('  SGLang：base 也算一个槽，所以 8 槽实际只给 7 个 adapter；drainer 让剩余最少的 running adapter 只收')
    print(f'          max_new ≤ 剩余 × {DRAIN_TOL} 的新请求，它自然跑空后把槽让给等得最久的 adapter。')
    print('          base（lora_id=None）也在 running_adapter_ids 里，同样可能被选中排空——表里 base p99 上涨就是这个。')
    print('  max_loras=1 是 vLLM 默认值：多 adapter 生产部署第一件事就是把它调到和热门 adapter 数相当。')


# --------------------------------------------------------------------------- 10. 前缀缓存隔离
def vllm_extra_key(lora):
    """kv_cache_utils.py:589–607。lora = (name, path) 或 None。"""
    return () if lora is None else (('lora', lora[0], lora[1]),)


def sgl_lora_id(lora, from_cli, rand):
    """lora_registry.py:37–54：--lora-paths 启动加载用 uuid5(name\\0path)，动态 load_lora_adapter 用 uuid4。"""
    if lora is None:
        return None
    if from_cli:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{lora[0]}\0{lora[1]}'))
    return str(uuid.UUID(int=rand.getrandbits(128), version=4))


def block_hash(parent, tokens, extra):
    h = hashlib.sha256(repr((parent, tuple(tokens), extra)).encode()).hexdigest()
    return h[:12]


def scenario_prefix():
    prompt = list(range(64))
    A1, A2, B1 = ('sql', '/ckpt/sql-v1'), ('sql', '/ckpt/sql-v2'), ('sql-new', '/ckpt/sql-v1')
    rand = random.Random(0)
    print('== 同一段 system prompt，第一块的哈希（只有 extra key 不同）')
    print(f'  {"情况":34s} {"vLLM key":>12s} {"SGLang(CLI)":>12s} {"SGLang(动态)":>12s}')
    cases = [('base', None), ('sql → v1', A1), ('sql 重新指向 v2', A2), ('改名 sql-new → v1', B1)]
    seen = {}
    for name, l in cases:
        v = block_hash(None, prompt[:16], vllm_extra_key(l))
        sc = block_hash(None, prompt[:16], sgl_lora_id(l, True, rand))
        sd = block_hash(None, prompt[:16], sgl_lora_id(l, False, rand))
        seen[name] = (v, sc, sd)
        print(f'  {name:34s} {v:>12s} {sc:>12s} {sd:>12s}')
    print()
    print('  判读：')
    print('  - 重新指向（同名换路径）：vLLM key 带 path，不会命中旧 KV；SGLang 两种 id 也都变了。三者都安全。')
    print('  - 改名但权重一样：vLLM 和 SGLang 都不复用——KV 白算一遍，但不会错。')
    print('  - 路径不变、文件被原地覆盖（训练脚本直接写回同一目录）：vLLM key 不变，会命中旧 KV；')
    print('    SGLang 用 CLI 加载的 uuid5 也不变；只有动态加载的 uuid4 每次都不同。上线新权重请换路径或改名。')
    print()
    print('== 多机 / 重启：同一个 adapter 的 key 是否稳定（影响跨实例的 KV 复用，比如 LMCache、PD 的前缀路由）')
    r1, r2 = random.Random(1), random.Random(2)
    rows = [('vLLM (name, path)', vllm_extra_key(A1) == vllm_extra_key(A1)),
            ('SGLang --lora-paths uuid5', sgl_lora_id(A1, True, r1) == sgl_lora_id(A1, True, r2)),
            ('SGLang 动态加载 uuid4', sgl_lora_id(A1, False, r1) == sgl_lora_id(A1, False, r2))]
    for n, same in rows:
        print(f'  {n:28s} 两台机器一致：{"是" if same else "否"}')
    print('  lora_int_id 不进哈希：vLLM 的 int id 只用来找 GPU 槽，同一 adapter 在两台机器上 id 不同也不影响命中。')


# --------------------------------------------------------------------------- 11. SGLang 对照
def scenario_sglang():
    rows = [
        ('结构化输出后端', 'auto：逐请求 xgrammar → guidance/outlines；引擎锁定首个', '--grammar-backend 启动定死，默认 xgrammar'),
        ('mask 与前向重叠', 'execute_model(non_block) 后在 engine core 算 bitmask', 'overlap 下 delay_sample：前向先发，CPU 推进语法、填 mask 后再采样'),
        ('并行填 mask', '> 128 请求且无投机：16 个一批，min(cpu//2,8) 线程', 'fill_vocab_mask_batched 一次填整批'),
        ('编译超时', f'仅正则 {REGEX_TIMEOUT_S} s', f'轮询 {SGL_MAX_POLL} 次 × max(5 ms, 步长)'),
        ('投机 + 语法', 'rollback 上限 = num_speculative_tokens；逐 draft 生成 k+1 行', f'MAX_ROLLBACK_TOKENS={SGL_ROLLBACK}'),
        ('jump-forward', '无（只剩注释）', 'try_jump_forward 仍在，调度路径没调用'),
        ('LoRA 槽', 'max_loras=1，base 不占槽', 'max_loras_per_batch=8，base 占一个槽'),
        ('LoRA 准入', 'running 里所有 adapter 占槽，满了 skip', 'validate_lora_batch + pinned 预留 + drainer'),
        ('LoRA 显存', 'max_loras × max_lora_rank 预分配', 'mem_pool 按 max_lora_rank 预分配，lru/fifo 驱逐，base 最后驱逐'),
        ('LoRA kernel', 'Punica 风格 Triton shrink/expand', 'csgmv（默认）/ triton / torch_native'),
        ('前缀隔离', 'extra key ("lora", name, path)', 'extra_key 拼 lora_id（uuid4 / uuid5）'),
    ]
    print(f'  {"维度":14s} | {"vLLM":52s} | SGLang')
    for a, b, c in rows:
        print(f'  {a:14s} | {b:52s} | {c}')


SCENARIOS = dict(mask=scenario_mask, overlap=scenario_overlap, compile=scenario_compile, spec=scenario_spec,
                 auto=scenario_auto, jump=scenario_jump, loramem=scenario_loramem, punica=scenario_punica,
                 sched=scenario_sched, prefix=scenario_prefix, sglang=scenario_sglang)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='mask')
    SCENARIOS[ap.parse_args().scenario]()
