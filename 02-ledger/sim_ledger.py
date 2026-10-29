"""推理账本：一张卡上权重、激活、CUDA Graph、KV 各占多少，最多放多少 token，decode 的理论上限是多少。

用法：
    python3 sim_ledger.py --model qwen3-8b --gpu h100-sxm
    python3 sim_ledger.py --model llama3.1-70b --gpu h100-sxm --tp 4 --kv-dtype fp8
    python3 sim_ledger.py --model deepseek-v3 --gpu h200 --tp 8 --weight-dtype fp8

扣减顺序对齐主干 vllm/v1/worker/gpu_worker.py 的 determine_available_memory：
    requested   = 总显存 x gpu_memory_utilization（注意是总显存，不是空闲显存）
    available   = requested - non_kv_cache_memory - cudagraph_estimate
    non_kv      = 权重 + profile_run 的激活峰值 + 非 torch 显存（NCCL、cuBLAS workspace 等）
    cudagraph   = profile_cudagraph_memory() 的估计，VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1（默认）时扣除

对照方法：用同样的模型和 --gpu-memory-utilization 启动 vLLM，启动日志会打印
    Available KV cache memory: X GiB
    GPU KV cache size: N tokens, Maximum concurrency for L tokens per request: C x
本脚本的 kv_gib、kv_tokens 应在 10% 以内；偏差大时先调 --act-gib 和 --graph-gib，
这两个值在日志里能找到（profile 的峰值激活、Graph capturing finished ... took X GiB）。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.specs import GPUS, MODELS  # noqa: E402

DTYPE = {'bf16': 2, 'fp16': 2, 'fp8': 1, 'int4': 0.5}
GiB = 1 << 30
BLOCK = 16  # DEFAULT_BLOCK_SIZE


def ledger(model, gpu, tp=1, util=0.92, weight_dtype='bf16', kv_dtype='bf16', avg_ctx=2048,
           act_gib=None, nontorch_gib=0.6, graph_gib=0.5, max_batched_tokens=8192, total_gib=None):
    # 厂商标称的 "80GB" 实际接近 80 GiB，驱动扣掉保留区后 torch 看到的 total_memory 再少 1% 左右。
    # 要精确对账，用 python -c "import torch;print(torch.cuda.mem_get_info()[1]/2**30)" 的值传 --total-gib。
    total = (total_gib if total_gib else gpu.mem_gb * 0.99) * GiB
    requested = total * util
    w_bytes = model.params_b * 1e9 * DTYPE[weight_dtype] / tp
    if act_gib is None:
        # profile_run 用 max_num_batched_tokens 个 token 做一次 dummy 前向。粗估峰值为
        # MLP 中间激活 + logits 两大头：tokens x hidden x 2B x 8，再加采样阶段 logits（最多 max_num_seqs 行）。
        act = max_batched_tokens * model.hidden * 2 * 8 / tp + 1024 * model.vocab * 4
        act_gib = act / GiB
    non_kv = w_bytes + act_gib * GiB + nontorch_gib * GiB
    kv_pool = requested - non_kv - graph_gib * GiB

    # MLA 的 latent 在 TP 下每卡都存一份完整副本；MHA/GQA 按 kv_heads 切分，kv_heads < tp 时复制
    kv_tok = model.kv_bytes_per_token(DTYPE[kv_dtype])
    if not model.mla_kv_rank:
        kv_tok /= min(tp, model.kv_heads)
    # vLLM 按块分配：块数向下取整，再乘回块大小得到 token 数
    blocks = int(kv_pool // (kv_tok * BLOCK)) if kv_pool > 0 else 0
    kv_tokens = blocks * BLOCK

    # decode 一步把本卡权重读一遍，再读 batch 内全部 KV；按带宽算理论下限时延。
    # MoE 只按激活参数计，相当于 batch 很小、专家几乎不重叠，batch 变大后会偏乐观，第 11 章细算。
    active_w = model.active_b * 1e9 * DTYPE[weight_dtype] / tp
    rows = []
    for bs in (1, 8, 32, 128, 256, 512):
        if bs * avg_ctx > kv_tokens:
            break
        bytes_step = active_w + bs * avg_ctx * kv_tok
        t = bytes_step / (gpu.bw_tbs * 1e12)
        flops = 2 * model.active_b * 1e9 / tp * bs
        rows.append((bs, t * 1e3, bs / t, flops / bytes_step))
    return dict(requested_gib=requested / GiB, weights_gib=w_bytes / GiB, act_gib=act_gib,
                nontorch_gib=nontorch_gib, graph_gib=graph_gib, kv_gib=kv_pool / GiB,
                kv_bytes_per_token=kv_tok, blocks=blocks, kv_tokens=kv_tokens,
                max_reqs=kv_tokens // avg_ctx, decode=rows, ridge=gpu.ridge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-8b', choices=MODELS)
    ap.add_argument('--gpu', default='h100-sxm', choices=GPUS)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--util', type=float, default=0.92, help='--gpu-memory-utilization，主干默认 0.92')
    ap.add_argument('--weight-dtype', default='bf16', choices=DTYPE)
    ap.add_argument('--kv-dtype', default='bf16', choices=DTYPE)
    ap.add_argument('--avg-ctx', type=int, default=2048, help='每个请求平均占用的上下文长度')
    ap.add_argument('--max-batched-tokens', type=int, default=8192,
                    help='决定 profile_run 的激活峰值；H100/H200 在线服务默认 8192')
    ap.add_argument('--act-gib', type=float, help='用实测激活峰值覆盖粗估')
    ap.add_argument('--nontorch-gib', type=float, default=0.6)
    ap.add_argument('--graph-gib', type=float, default=0.5, help='CUDA Graph 显存估计，--enforce-eager 时填 0')
    ap.add_argument('--total-gib', type=float, help='torch 报告的单卡总显存，默认按标称容量的 99%% 估')
    a = ap.parse_args()
    m, g = MODELS[a.model], GPUS[a.gpu]
    L = ledger(m, g, a.tp, a.util, a.weight_dtype, a.kv_dtype, a.avg_ctx,
               a.act_gib, a.nontorch_gib, a.graph_gib, a.max_batched_tokens, a.total_gib)
    print(f'{m.name} on {a.tp} x {g.name}  weight={a.weight_dtype} kv={a.kv_dtype} util={a.util}')
    print(f'  requested       {L["requested_gib"]:8.2f} GiB  (总显存 x util)')
    print(f'  - 权重          {L["weights_gib"]:8.2f} GiB')
    print(f'  - 激活峰值      {L["act_gib"]:8.2f} GiB')
    print(f'  - 非 torch      {L["nontorch_gib"]:8.2f} GiB')
    print(f'  - CUDA Graph    {L["graph_gib"]:8.2f} GiB')
    print(f'  = KV 可用       {L["kv_gib"]:8.2f} GiB  ← 对照日志 Available KV cache memory')
    print(f'  每 token KV     {L["kv_bytes_per_token"] / 1024:8.1f} KiB（每卡）')
    print(f'  KV 块 / token   {L["blocks"]:8d} 块 x {BLOCK} = {L["kv_tokens"]:,} token  ← 对照 GPU KV cache size')
    print(f'  按 {a.avg_ctx} token/请求  最多并发 {L["max_reqs"]} 个请求')
    print(f'  Roofline 拐点   {L["ridge"]:8.0f} FLOP/Byte')
    if L['kv_tokens'] == 0:
        print('  KV 池为负：权重放不下，增大 --tp、降低权重精度或调高 --util')
        return
    print('  decode 带宽下限（只计 HBM 读取，不含 kernel 和调度开销）')
    print('    batch   step(ms)   tok/s     算术强度(FLOP/B)')
    for bs, ms, tps, ai in L['decode']:
        print(f'    {bs:5d}   {ms:8.2f}   {tps:8.0f}   {ai:8.1f}')


if __name__ == '__main__':
    main()
