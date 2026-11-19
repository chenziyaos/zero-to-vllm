"""按主干 determine_available_memory 的顺序，把一次启动的显存账从头算一遍。

不需要 GPU。默认输入取自 01-first-server/sample_startup.log（H100 80GB SXM + Qwen3-8B BF16，
--gpu-memory-utilization 0.92，max_num_seqs 1024，max_num_batched_tokens 8192），
每一步都打印和启动日志对应的那一行，方便逐行对账。

用法：
    python3 sim_profile_run.py                       # 用示意日志的数字
    python3 sim_profile_run.py --log my_startup.log  # 从真实日志里抽数字再推一遍
    python3 sim_profile_run.py --util 0.95 --max-model-len 32768
    python3 sim_profile_run.py --sglang              # 追加 SGLang 默认 mem_fraction_static 的推导

对应源码（vllm 主干 0d6f92a）：
    vllm/v1/worker/utils.py                 request_memory
    vllm/utils/mem_utils.py                 MemorySnapshot / memory_profiling
    vllm/v1/worker/gpu_worker.py            determine_available_memory 与 kv 建议值
    vllm/v1/worker/gpu/cudagraph_utils.py   profile_cudagraph_memory / _extrapolate_full_graph_memory
    vllm/config/vllm.py                     _set_cudagraph_sizes
    vllm/v1/core/kv_cache_planning.py       null block 预留与 max_model_len 检查
    vllm/v1/core/kv_cache_utils.py          get_max_concurrency_for_kv_cache_config
SGLang 对照（主干 8fabf5e）：
    python/sglang/srt/arg_groups/memory_hook.py      handle_gpu_memory_settings / reserve_for_graph_mb
    python/sglang/srt/mem_cache/kv_cache_configurator.py  _profile_available_bytes
"""
import argparse
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '01-first-server'))
from common.specs import MODELS  # noqa: E402

GiB = 1 << 30
MiB = 1 << 20
DTYPE = {'bf16': 2, 'fp16': 2, 'fp8': 1}

# gpu_worker.py 的 kv 建议值行（INFO），一行里同时给出 free/total、consumed、peak activation 和实际图显存
SUGGEST = re.compile(
    r'Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup\. Desired GPU memory utilization is '
    r'\(([\d.]+), ([\d.]+) GiB\)\. Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), '
    r'([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph memory')


def gib(x):
    return x / GiB


def vllm_capture_sizes(max_num_seqs, max_num_batched_tokens, decode_query_len=1, blackwell=False,
                       interactivity=False):
    """config/vllm.py _set_cudagraph_sizes 的默认分支（不含投机解码的分层网格）。"""
    default_max = 1024 if blackwell else 512
    max_size = min(max_num_seqs * decode_query_len * 2, default_max)
    if interactivity:
        sizes = list(range(1, 33))
    else:
        sizes = [1, 2, 4] + list(range(8, min(max_size + 1, 256), 8)) + list(range(256, max_size + 1, 16))
    return [s for s in sizes if s <= max_num_batched_tokens and s <= max_size]


def extrapolate_full(samples, n_graphs):
    """cudagraph_utils.py _extrapolate_full_graph_memory：只真捕获最大的两张 FULL 图，其余按第二张外推。"""
    if not samples:
        return 0
    per = max(samples[1], MiB) if len(samples) > 1 else 0
    return samples[0] + (n_graphs - 1) * per


def sglang_capture_sizes():
    """SGLang 默认（非投机解码）的 decode / prefill 捕获网格，H100 80GB、TP=1 档位。"""
    decode_max = 256
    decode = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16)) + list(range(512, decode_max + 1, 32))
    decode = sorted(set([d for d in decode if d <= decode_max] + [decode_max]))
    prefill_max = 8192
    prefill = (list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32))
               + list(range(576, 1025, 64)) + list(range(1280, 4097, 256)) + list(range(4608, prefill_max + 1, 512)))
    return decode, prefill


def from_log(path):
    from parse_startup_log import parse  # noqa: E402
    lines = open(path, errors='replace').read().splitlines()
    found, _, _, _ = parse(lines)
    out = {}
    for ln in lines:
        m = SUGGEST.search(ln)
        if m:
            free, total, util, req, consumed, peak, cg = map(float, m.groups())
            out.update(free_gib=free, total_gib=total, util=util, consumed_gib=consumed, peak_gib=peak,
                       cg_actual_gib=cg)
    if 'model_load' in found:
        out['weights_gib'] = float(found['model_load'][0])
    if 'graph_pool' in found:
        out['cg_actual_gib'], out['cg_est_gib'] = map(float, found['graph_pool'])
    if 'kv_avail_gib' in found:
        out['kv_logged_gib'] = float(found['kv_avail_gib'][0])
    if 'kv_tokens' in found:
        out['kv_tokens_logged'] = int(found['kv_tokens'][0].replace(',', ''))
        out['max_model_len'] = int(found['kv_tokens'][1].replace(',', ''))
    if 'max_batched_tokens' in found:
        out['max_batched_tokens'] = int(found['max_batched_tokens'][0])
    if 'mem_profile' in found:
        _, nonkv, torch_peak, consumed, weights = map(float, found['mem_profile'])
        out.update(consumed_gib=consumed, weights_gib=weights)
    if 'peak_gib' in out and 'cg_est_gib' in out:
        out['transient_gib'] = out['peak_gib'] - out['cg_est_gib']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', help='真实启动日志；有就优先用日志里的数字')
    ap.add_argument('--model', default='qwen3-8b', choices=MODELS)
    ap.add_argument('--kv-dtype', default='bf16', choices=DTYPE)
    ap.add_argument('--total-bytes', type=int, default=85_045_231_616, help='torch.cuda.mem_get_info()[1]')
    ap.add_argument('--free-gib', type=float, help='启动时的空闲显存，默认按整卡减 0.5 GiB（驱动上下文等）')
    ap.add_argument('--util', type=float, default=0.92)
    ap.add_argument('--weights-gib', type=float, default=15.27)
    ap.add_argument('--nontorch-gib', type=float, default=0.62)
    ap.add_argument('--transient-gib', type=float, default=1.43, help='profile_run 的 torch 峰值减去常驻分配')
    ap.add_argument('--cg-est-gib', type=float, default=0.71)
    ap.add_argument('--cg-actual-gib', type=float, default=0.68)
    ap.add_argument('--no-cg-estimate', action='store_true', help='模拟 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0')
    ap.add_argument('--block-size', type=int, default=16)
    ap.add_argument('--max-model-len', type=int, default=40960)
    ap.add_argument('--max-num-seqs', type=int, default=1024)
    ap.add_argument('--max-batched-tokens', type=int, default=8192)
    ap.add_argument('--sglang', action='store_true')
    a = ap.parse_args()

    v = dict(total_gib=a.total_bytes / GiB, free_gib=a.free_gib or a.total_bytes / GiB - 0.5, util=a.util, weights_gib=a.weights_gib,
             consumed_gib=a.weights_gib + a.nontorch_gib, transient_gib=a.transient_gib,
             cg_est_gib=a.cg_est_gib, cg_actual_gib=a.cg_actual_gib, max_model_len=a.max_model_len,
             max_batched_tokens=a.max_batched_tokens)
    if a.log:
        got = from_log(a.log)
        print(f'从 {a.log} 抽到：' + ', '.join(f'{k}={got[k]}' for k in sorted(got)))
        v.update(got)
        print('日志里的 GiB 只保留两位小数，反推的 requested、KV、token 会有 0.01 GiB、几十 token 量级的取整偏差')
    total = a.total_bytes if not a.log else v['total_gib'] * GiB
    m = MODELS[a.model]

    print('\n[1] request_memory  (v1/worker/utils.py)')
    requested = math.ceil(total * v['util'])
    print(f'    requested = ceil({total:,.0f} x {v["util"]}) = {requested:,} B = {gib(requested):.2f} GiB')
    if v['free_gib'] * GiB < requested:
        print(f'    空闲 {v["free_gib"]:.2f} GiB < requested，启动会直接报 Free memory on device ... less than desired')
        return

    print('\n[2] memory_profiling  (utils/mem_utils.py)')
    consumed = v['consumed_gib'] * GiB
    transient = v['transient_gib'] * GiB
    non_kv = consumed + transient
    print(f'    total_consumed = 启动前 free - profile 后 free = {gib(consumed):.2f} GiB'
          f'（权重 {v["weights_gib"]:.2f} + 非 torch {v["consumed_gib"] - v["weights_gib"]:.2f}）')
    print(f'    transient_peak_headroom = torch 峰值 - torch 常驻 = {gib(transient):.2f} GiB')
    print(f'    non_kv_cache_memory = {gib(non_kv):.2f} GiB')

    print('\n[3] profile_cudagraph_memory  (v1/worker/gpu/cudagraph_utils.py)')
    sizes = vllm_capture_sizes(a.max_num_seqs, v['max_batched_tokens'])
    print(f'    捕获网格 {len(sizes)} 个尺寸：{sizes[:6]} ... {sizes[-4:]}  ← 日志 Capturing CUDA graphs ... {len(sizes)}/{len(sizes)}')
    print(f'    FULL 图只真捕获最大的 2 张，其余 {len(sizes) - 1} 张按第二张外推，单张下限 1 MiB：')
    demo = [int(0.30 * GiB), int(4 * MiB)]
    print(f'      例：samples=[{demo[0] / MiB:.0f} MiB, {demo[1] / MiB:.0f} MiB] → '
          f'{demo[0] / MiB:.0f} + {len(sizes) - 1} x {demo[1] / MiB:.0f} = {extrapolate_full(demo, len(sizes)) / MiB:.0f} MiB（示意数字）')
    cg_est = v['cg_est_gib'] * GiB
    cg_applied = 0 if a.no_cg_estimate else cg_est
    print(f'    估计 {gib(cg_est):.2f} GiB，{"不扣（ESTIMATE_CUDAGRAPHS=0）" if a.no_cg_estimate else "从 KV 里扣掉"}')

    print('\n[4] Available KV cache memory  (gpu_worker.py)')
    avail = requested - non_kv - cg_applied
    print(f'    {gib(requested):.2f} - {gib(non_kv):.2f} - {gib(cg_applied):.2f} = {gib(avail):.2f} GiB'
          + (f'   日志：{v["kv_logged_gib"]:.2f}' if 'kv_logged_gib' in v else ''))
    peak_act = transient + cg_applied

    print('\n[5] 块与 token  (kv_cache_planning.py / kv_cache_utils.py)')
    kv_tok = m.kv_bytes_per_token(DTYPE[a.kv_dtype])
    page = kv_tok * a.block_size
    blocks = int(avail // page)
    per_req = math.ceil(v['max_model_len'] / a.block_size)
    conc = blocks / per_req
    tokens = int(conc * v['max_model_len'])
    print(f'    每块 {a.block_size} token x {kv_tok / 1024:.0f} KiB = {page:,.0f} B（36 层合计）')
    print(f'    num_blocks = {gib(avail):.2f} GiB // page = {blocks:,}，其中 1 块是 null block，可用 {blocks - 1:,}')
    need = per_req * page
    if avail - page < need:
        print(f'    一条 {v["max_model_len"]} token 的请求要 {gib(need):.2f} GiB，装不下 → 启动报 To serve at least one request ...')
    print(f'    并发 = {blocks:,} / cdiv({v["max_model_len"]}, {a.block_size}) = {conc:.2f}x')
    print(f'    GPU KV cache size = int({conc:.4f} x {v["max_model_len"]}) = {tokens:,} tokens'
          + (f'   日志：{v["kv_tokens_logged"]:,}' if 'kv_tokens_logged' in v else ''))
    print(f'    （日志的 token 数含 null block 的 {a.block_size} 个槽位；调度器真正能分出去的是 {(blocks - 1) * a.block_size:,}）')

    if cg_est > 0 and not a.no_cg_estimate:
        print('\n[6] CUDA graph util 提示行  (gpu_worker.py)')
        d = cg_est / total
        print(f'    delta = {gib(cg_est):.2f} / {gib(total):.2f} = {d:.6f}；等价旧口径 {round(v["util"] - d, 4):.4f}，'
              f'保持旧 KV 需调到 {min(round(v["util"] + d, 4), 1.0):.4f}')

    print('\n[7] --kv-cache-memory 建议值  (gpu_worker.py，捕获真实图之后)')
    buf = 150 * MiB
    cg_act = v['cg_actual_gib'] * GiB
    non_kv2 = consumed + peak_act + cg_act
    to_req = int(requested) - non_kv2 - buf
    to_gpu = v['free_gib'] * GiB - non_kv2 - buf
    print(f'    non_kv = consumed {gib(consumed):.2f} + peak_activation {gib(peak_act):.2f}（含图估计）'
          f' + 实际图 {gib(cg_act):.2f} = {gib(non_kv2):.2f} GiB')
    print(f'    贴着 requested：{int(to_req):,} B（{gib(to_req):.2f} GiB）')
    print(f'    贴着整卡空闲：  {int(to_gpu):,} B（{gib(to_gpu):.2f} GiB）')
    print(f'    比当前 KV 少 {gib(avail - to_req):.2f} GiB：图显存在 peak_activation 里算过一次、这里又加了一次，再扣 150 MiB 缓冲')
    print('    注意：提示文案写的是 --kv-cache-memory=，正式参数名是 --kv-cache-memory-bytes；'
          '前者能用只因 argparse 默认接受唯一前缀，部署脚本里写全名')

    if a.sglang:
        print('\n[8] SGLang 默认 mem_fraction_static（memory_hook.py，H100 TP=1，chunked_prefill 8192，decode max_bs 256）')
        gpu_mem_mib = 81559  # nvidia-smi --query-gpu=memory.total，H100 80GB SXM
        decode, prefill = sglang_capture_sizes()
        act = max(v['max_batched_tokens'], 2048) * 1.5
        par = 1 * 1 / 8 * 1024
        graph = 256 * 2 + len(prefill) * 8
        reserved = 512 + act + par + graph
        if gpu_mem_mib > 60 * 1024:
            reserved = max(reserved, 10 * 1024)
        mfs = round((gpu_mem_mib - reserved) / gpu_mem_mib, 3)
        print(f'    reserved = 512 + {act:.0f}（激活）+ {par:.0f}（并行）+ {graph}（decode 图 256x2 + prefill 图 {len(prefill)}x8）'
              f' = {reserved:,.0f} MiB')
        print(f'    mem_fraction_static = round(({gpu_mem_mib} - {reserved:.0f}) / {gpu_mem_mib}, 3) = {mfs:.3f}')
        print(f'    decode 网格 {len(decode)} 个尺寸，prefill 网格 {len(prefill)} 个尺寸')
        pre = v['free_gib']
        after = pre - v['weights_gib']
        slack = pre * (1 - mfs)
        rest = after - slack
        print(f'    KV 预算 = 加载后空闲 {after:.2f} - 加载前空闲 {pre:.2f} x (1 - {mfs:.3f}) = {rest:.2f} GiB'
              f'（未计加载期间新增的非 torch 显存）')
        print(f'    ≈ {int(rest * GiB / kv_tok):,} token；vLLM 同卡 {tokens:,} token')


if __name__ == '__main__':
    main()
