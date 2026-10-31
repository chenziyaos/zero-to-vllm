#!/usr/bin/env bash
# VLLM-17 调参实验：先看引擎替你选了什么，再按"显存 → 预算 → 并发 → 准入"的顺序扫。单卡即可。
#
# 用法：
#   bash sweep.sh defaults                  # 起一次默认实例，抓 max_num_batched_tokens / max_num_seqs / 捕获网格 / KV 的日志行
#   bash sweep.sh util                      # gpu_memory_utilization 0.85 → 0.97 逐档起服，记 Available KV cache memory 与 token 数
#   bash sweep.sh grid                      # vllm bench sweep serve：serve_params.json × bench_params.json 笛卡尔积
#   bash sweep.sh workload                  # vllm bench sweep serve_workload：从串行到满并发，画延迟-吞吐曲线
#   bash sweep.sh goodput [TTFT_MS] [TPOT_MS]  # 固定 request_rate，用 --goodput 比较不同预算的 SLO 达标率
#   bash sweep.sh startup                   # vllm bench sweep startup：O0 / O2 / enforce-eager 的冷热启动时间
#   bash sweep.sh queue [LIMIT]             # --max-num-queued-tokens 打开后过载压测，看 503 与 TTFT p99
#   bash sweep.sh plot                      # vllm bench sweep plot / plot_pareto
#   bash sweep.sh sgl_defaults              # SGLang 对照：默认启动，抓 mem_fraction_static / chunked_prefill_size / cuda graph bs
#   bash sweep.sh sgl_grid                  # SGLang 对照：chunked_prefill_size × max_running_requests 网格，用同一个 vllm bench serve 压
#
# 环境变量：MODEL（默认 Qwen/Qwen3-8B）、OUT（默认 results）、MAXLEN（默认 8192）、PORT（默认 8000）、
#           RATE（goodput / queue 的 request_rate，默认 10）、NPROMPTS（默认 600）。
# 日志里要找的几行（vllm 0d6f92a / sglang 8fabf5e）：
#   "Defaulting max_num_batched_tokens to"          arg_utils.py，DEBUG 级别，需 VLLM_LOGGING_LEVEL=DEBUG
#   "Capturing CUDA graphs ... N/N"                  捕获网格个数；H100 默认 51
#   "Available KV cache memory: X GiB"               gpu_worker.py:776
#   "GPU KV cache size: N tokens"                    kv_cache_utils.py
#   "max_num_batched_tokens (X) exceeds max_num_seqs * max_model_len"   scheduler.py:347 的 warning
#   SGLang："mem_fraction_static=" / "chunked_prefill_size=" / "max_total_num_tokens=" 出现在 server_args 打印与 KV 初始化日志
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUT=${OUT:-results}
MAXLEN=${MAXLEN:-8192}
PORT=${PORT:-8000}
RATE=${RATE:-10}
NPROMPTS=${NPROMPTS:-600}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"

# 06 章混合负载的近似：random 数据集，输入 1500±80%，输出 300±60%
BENCH="vllm bench serve --model $MODEL --backend vllm --endpoint /v1/completions --port $PORT \
  --dataset-name random --random-input-len 1500 --random-output-len 300 --random-range-ratio 0.8 \
  --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99"

wait_ready() {
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "server not ready" >&2; return 1
}

serve_bg() {  # serve_bg LOG ARGS...
  local log=$1; shift
  vllm serve "$MODEL" --port "$PORT" --max-model-len "$MAXLEN" "$@" >"$log" 2>&1 &
  echo $! >"$OUT/.pid"
  wait_ready
}

stop() { [ -f "$OUT/.pid" ] && kill "$(cat "$OUT/.pid")" 2>/dev/null || true; sleep 5; }

case "${1:-}" in
defaults)
  VLLM_LOGGING_LEVEL=DEBUG serve_bg "$OUT/defaults.log"
  stop
  grep -E "Defaulting max_num_batched_tokens|Capturing CUDA graphs|Available KV cache memory|GPU KV cache size|Maximum concurrency" \
    "$OUT/defaults.log" | tail -n 8
  echo "对照：python3 $HERE/sim_tuning.py --scenario defaults"
  ;;
util)
  for u in 0.85 0.88 0.90 0.92 0.94 0.95 0.97; do
    serve_bg "$OUT/util_$u.log" --gpu-memory-utilization "$u" || { echo "util $u 起不来"; stop; continue; }
    stop
    echo "util=$u  $(grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$OUT/util_$u.log" | tail -1)  \
$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$OUT/util_$u.log" | tail -1)"
  done
  echo "每 0.01 应接近 0.79 GiB / 5.8k token（H100 + Qwen3-8B）；对照 sim_tuning.py --scenario mem"
  ;;
grid)
  vllm bench sweep serve \
    --serve-cmd "vllm serve $MODEL --port $PORT --max-model-len $MAXLEN" \
    --bench-cmd "$BENCH --num-prompts $NPROMPTS --request-rate $RATE" \
    --serve-params "$HERE/serve_params.json" \
    --bench-params "$HERE/bench_params.json" \
    --num-runs 3 --warmup-num-prompts 50 --continue-on-error \
    --output-dir "$OUT/grid" --experiment-name budget_x_seqs
  ;;
workload)
  vllm bench sweep serve_workload \
    --serve-cmd "vllm serve $MODEL --port $PORT --max-model-len $MAXLEN" \
    --bench-cmd "$BENCH --num-prompts $NPROMPTS" \
    --workload-var max_concurrency --workload-iters 8 \
    --serve-params "$HERE/serve_params.json" \
    --num-runs 1 --output-dir "$OUT/workload" --experiment-name curve
  ;;
goodput)
  TTFT=${2:-2000}; TPOT=${3:-30}
  for b in 1024 2048 4096 8192 16384; do
    serve_bg "$OUT/gp_$b.log" --max-num-batched-tokens "$b"
    $BENCH --num-prompts "$NPROMPTS" --request-rate "$RATE" --goodput "ttft:$TTFT" "tpot:$TPOT" \
      --save-result --result-dir "$OUT" --result-filename "gp_$b.json" | grep -E "Request goodput|P99 TTFT|P99 ITL|Output token throughput"
    stop
  done
  echo "对照：python3 $HERE/sim_tuning.py --scenario budget --rate $RATE --slo-ttft $TTFT --slo-tpot $TPOT"
  ;;
startup)
  cat >"$OUT/startup_params.json" <<EOF
[
  {"_benchmark_name": "O0", "model": "$MODEL", "optimization_level": 0},
  {"_benchmark_name": "O2", "model": "$MODEL", "optimization_level": 2},
  {"_benchmark_name": "eager", "model": "$MODEL", "enforce_eager": true}
]
EOF
  vllm bench sweep startup --startup-cmd "vllm bench startup --model $MODEL" \
    --serve-params "$OUT/startup_params.json" \
    --output-dir "$OUT/startup" --experiment-name opt_levels
  echo "热启动复用 VLLM_CACHE_ROOT（默认 ~/.cache/vllm）；容器里把这个目录挂盘或打进镜像"
  ;;
queue)
  LIMIT=${2:-36000}
  for lim in none "$LIMIT"; do
    extra=(); [ "$lim" != none ] && extra=(--max-num-queued-tokens "$lim")
    serve_bg "$OUT/queue_$lim.log" "${extra[@]}"
    $BENCH --num-prompts "$NPROMPTS" --request-rate "$((RATE * 2))" | grep -E "Successful requests|Failed requests|P99 TTFT|Median TTFT"
    stop
  done
  echo "被拒的请求返回 HTTP 503；/metrics 里看 admission rejection 计数（loggers.py ADMISSION_REASONS）"
  ;;
plot)
  vllm bench sweep plot "$OUT/grid" --var-x max_concurrency --row-by random_input_len --col-by max_num_batched_tokens || true
  vllm bench sweep plot_pareto "$OUT/workload" || true
  ;;
sgl_defaults)
  python3 -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --context-length "$MAXLEN" >"$OUT/sgl_defaults.log" 2>&1 &
  echo $! >"$OUT/.pid"; wait_ready; stop
  grep -oE "mem_fraction_static=[0-9.]+|chunked_prefill_size=[0-9-]+|max_running_requests=[0-9A-Za-z]+|max_total_num_tokens=[0-9]+" \
    "$OUT/sgl_defaults.log" | sort -u
  echo "对照：python3 $HERE/sim_tuning.py --scenario mem（H100 应为 0.830）"
  ;;
sgl_grid)
  for c in 2048 4096 8192; do
    for r in 128 256; do
      python3 -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --context-length "$MAXLEN" \
        --chunked-prefill-size "$c" --max-running-requests "$r" >"$OUT/sgl_${c}_${r}.log" 2>&1 &
      echo $! >"$OUT/.pid"; wait_ready
      ${BENCH/--backend vllm/--backend openai} --num-prompts "$NPROMPTS" --request-rate "$RATE" \
        --goodput ttft:2000 tpot:30 | grep -E "Request goodput|P99 TTFT|P99 ITL"
      stop
    done
  done
  echo "SGLang 健康指标：日志 #queue-req 100–2000、token usage > 0.9；频繁 retract 调 --schedule-conservativeness 1.3"
  ;;
*)
  sed -n '2,16p' "$0"; exit 1 ;;
esac
