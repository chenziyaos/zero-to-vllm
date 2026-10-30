#!/usr/bin/env bash
# VLLM-11 GPU 实验：同一台 8 卡机器，换并行方式起服务，看分组日志、KV 容量、EPLB balancedness 和吞吐。
#
# 用法：
#   bash serve_dp_ep.sh tp8          # 纯 TP8：基线
#   bash serve_dp_ep.sh dpep         # -dp 8 -ep：注意力 DP + 专家 EP（MoE 模型）
#   bash serve_dp_ep.sh eplb         # 在 dpep 上加 --enable-eplb + 冗余专家，打开 balancedness 日志
#   bash serve_dp_ep.sh dbo          # 在 dpep 上加 --enable-dbo（microbatch overlap）
#   bash serve_dp_ep.sh dcp          # -tp 8 -dcp 8（MLA）或 -tp 8 -dcp 2（GQA，需 tp > kv_heads）
#   bash serve_dp_ep.sh pp           # -pp 2 -tp 4，看 "Hidden layers were unevenly partitioned" 日志
#   bash serve_dp_ep.sh bench        # 对已经起好的服务跑 vllm bench serve（随机 1024/256，并发 256）
#   bash serve_dp_ep.sh sgl_tp8|sgl_dpep|sgl_eplb|sgl_dcp   # SGLang 对照（需要另装 sglang）
#
# 环境变量：MODEL（默认 deepseek-ai/DeepSeek-V3）、OUT、PORT、GMU、REDUNDANT（默认 32）、DCP（默认 8）。
# 日志里要找的几行（行号对应 vllm 0d6f92a）：
#   "Hidden layers were unevenly partitioned: [...]"          distributed/utils.py:230
#   "EPLB step: %d for model %s: avg_tokens=..., balancedness=..."   distributed/eplb/eplb_state.py:635（log_balancedness=true）
#   "num_gpus % num_nodes != 0, not using hierarchical rearrangement algorithm."   eplb_state.py:850
#   "Offline data parallel mode is not supported/useful for dense models."         config/parallel.py:1050
#   "exceeds the maximum supported value" / "requires --tensor-parallel-size"     config/model.py:1585–1613（DCP）
#   "GPU KV cache size: N tokens"                                v1/core/kv_cache_utils.py:1148（每个 DP rank 各打一行）
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V3}
OUT=${OUT:-results}
PORT=${PORT:-8000}
GMU=${GMU:-0.90}
REDUNDANT=${REDUNDANT:-32}
DCP=${DCP:-8}
PATTERN='unevenly partitioned|EPLB step|hierarchical|KV cache size|Maximum concurrency|world_size|data_parallel|DP rank|dcp|DBO|ubatch|all2all|Using .* backend'

mkdir -p "$OUT"

serve() {
  local tag=$1; shift
  echo "== vllm serve $MODEL $*"
  vllm serve "$MODEL" --port "$PORT" --gpu-memory-utilization "$GMU" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered "$PATTERN" || true
}

sgl() {
  local tag=$1; shift
  echo "== sglang.launch_server --model-path $MODEL $*"
  python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --mem-fraction-static "$GMU" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered 'KV Cache|max_total_num_tokens|attn_dp|ep_size|eplb|dcp|chunked_prefill_size' || true
}

case "${1:-}" in
  tp8)   serve tp8   -tp 8 ;;
  dpep)  serve dpep  -dp 8 -ep ;;
  eplb)  serve eplb  -dp 8 -ep --enable-eplb \
           --eplb-config "{\"num_redundant_experts\":$REDUNDANT,\"log_balancedness\":true,\"log_balancedness_interval\":100}" ;;
  dbo)   serve dbo   -dp 8 -ep --enable-dbo ;;
  dcp)   serve dcp   -tp 8 -dcp "$DCP" ;;
  pp)    serve pp    -pp 2 -tp 4 ;;
  bench)
    vllm bench serve --model "$MODEL" --port "$PORT" --dataset-name random \
      --random-input-len 1024 --random-output-len 256 --num-prompts 2048 --max-concurrency 256 \
      --save-result --result-dir "$OUT" 2>&1 | tee "$OUT/bench.log" | \
      grep -E "Output token throughput|Mean TTFT|Mean TPOT|P99" || true ;;
  sgl_tp8)  sgl sgl_tp8  --tp 8 ;;
  sgl_dpep) sgl sgl_dpep --tp 8 --attn-dp-size 8 --ep-size 8 ;;
  sgl_eplb) sgl sgl_eplb --tp 8 --attn-dp-size 8 --ep-size 8 --enable-eplb --ep-num-redundant-experts "$REDUNDANT" ;;
  sgl_dcp)  sgl sgl_dcp  --tp 8 --dcp-size "$DCP" ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
