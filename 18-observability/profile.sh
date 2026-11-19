#!/usr/bin/env bash
# 18 章配套：指标抓取与 profiling 的命令清单。
# 每段都是独立的，按需复制；不要整份直接跑。
# 版本：vLLM 0d6f92a（docs/contributing/profiling.md），SGLang 8fabf5e（benchmark_and_profiling.mdx）
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-8B}
VLLM=${VLLM:-http://127.0.0.1:8000}
SGL=${SGL:-http://127.0.0.1:30000}

case "${1:-help}" in

# ---------------------------------------------------------------- 指标
metrics)
  # vLLM 默认就开 /metrics；SGLang 要 --enable-metrics
  curl -s "$VLLM/metrics" | grep -E '^vllm:(time_to_first_token|inter_token_latency|request_time_per_output_token)_seconds_(count|sum)'
  curl -s "$VLLM/metrics" | grep -E '^vllm:(prefix_cache_(queries|hits)_total|kv_cache_usage_perc|num_requests_(running|waiting))'
  # 两次抓取算窗口 p50/p99、命中率、computed vs total prompt tok/s
  python3 "$(dirname "$0")/scrape_metrics.py" --url "$VLLM/metrics" --interval 30
  ;;

buckets)
  # 低延迟场景把 TTFT / ITL 桶加密（键是 buckets.py 的族名，ITL 族同时作用于 TPOT 直方图）
  vllm serve "$MODEL" --custom-histogram-buckets '{
    "time_to_first_token": [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.5,2.0,3.0,5.0,10.0],
    "inter_token_latency": [0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5,1.0,2.5]}'
  # SGLang 对应：--bucket-time-to-first-token / --bucket-inter-token-latency / --bucket-e2e-request-latency
  ;;

# ---------------------------------------------------------------- torch profiler
torch-serve)
  # delay/max 在这里是默认值；也可以每次在 /start_profile 的 body 里覆盖
  vllm serve "$MODEL" --profiler-config '{
    "profiler": "torch", "torch_profiler_dir": "./vllm_profile",
    "warmup_iterations": 2, "active_iterations": 5}'
  ;;

torch-window)
  # 跳过 5000 个 worker step 后记 20 个；记满后仍要 /stop_profile 才写表，未 stop 再 start 返回 409
  curl -s -X POST "$VLLM/start_profile" -H 'Content-Type: application/json' -d '{
    "profile_prefix": "ch18_run1", "delay_iterations": 5000, "max_iterations": 20,
    "profiler_kwargs": {"record_shapes": true, "with_stack": true}}'
  echo "压测中……"; sleep "${SECS:-60}"
  curl -s -X POST "$VLLM/stop_profile"
  ;;

torch-bench)
  # bench 自己调 /start_profile 和 /stop_profile
  vllm bench serve --backend vllm --model "$MODEL" --dataset-name random \
    --random-input-len 2000 --random-output-len 200 --num-prompts 50 --profile
  ;;

# ---------------------------------------------------------------- nsys
nsys-serve)
  # profiler=cuda 让 /start_profile 变成 cudaProfilerStart；repeat 让每次 bench 各出一份报告
  VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
    --trace-fork-before-exec=true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end repeat \
    vllm serve "$MODEL" --profiler-config.profiler cuda
  ;;

nsys-stats)
  nsys stats --report cuda_gpu_kern_sum "${2:?report.nsys-rep}" | head -30
  ;;

# ---------------------------------------------------------------- proton
proton-serve)
  # CUDA graph 开着时必须 proton_graph_attribution=true 才能归因 replay 的 kernel
  vllm serve "$MODEL" --profiler-config '{
    "profiler": "proton", "proton_profiler_dir": "./proton_profile",
    "proton_output_format": "hatchet", "proton_hook": "triton",
    "proton_graph_attribution": true}'
  # 查看：proton-viewer -m time/ns proton_profile/<file>.hatchet
  ;;

# ---------------------------------------------------------------- SGLang 对照
sgl-torch)
  # num_steps 记满自动 stop（vLLM 的 max_iterations 记满后仍要手动 stop）
  curl -s -X POST "$SGL/start_profile" -H 'Content-Type: application/json' -d '{
    "start_step": 3, "num_steps": 10, "activities": ["CPU", "GPU"]}'
  ;;

sgl-nsys)
  nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o sgl_profile \
    python -m sglang.launch_server --model-path "$MODEL" --enable-metrics
  # 另一个终端：curl -X POST $SGL/start_profile -d '{"start_step":3,"num_steps":10,"activities":["CUDA_PROFILER"]}'
  ;;

*)
  sed -n '2,4p' "$0"
  echo "子命令：metrics buckets torch-serve torch-window torch-bench nsys-serve nsys-stats proton-serve sgl-torch sgl-nsys"
  ;;
esac
