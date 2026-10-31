#!/usr/bin/env bash
# VLLM-15 GPU 实验：单机两卡起 1P1D（GPU0 = P，GPU1 = D），对照同机 DP=2，看 ITL 尾巴、TTFT 和 NIXL 传输指标。
# 每个子命令起一个进程，前台运行；开三个终端分别起 P、D、proxy，第四个终端跑 bench。
#
# 用法：
#   bash serve_pd_nixl.sh colo                 # 同机基线：GPU0、GPU1 各一个普通实例（DP=2 由 proxy 轮询，见 bench colo）
#   bash serve_pd_nixl.sh p | d                # pull 模式（NixlConnector）：P 端口 8100 / side channel 5600，D 8200 / 5601
#   bash serve_pd_nixl.sh proxy                # tests/.../toy_proxy_server.py，端口 8192（P 返回后再发 D，串行）
#   bash serve_pd_nixl.sh push_p | push_d      # NixlPushConnector：P 的 engine_id 固定为 prefill-engine-001
#   bash serve_pd_nixl.sh push_proxy           # disagg_proxy_pushconnector_demo.py：P、D 同时发，D 提前向 P 登记
#   bash serve_pd_nixl.sh bidir_p | bidir_d    # bidirectional_kv_xfer=true，D 侧块留 decoder_kv_blocks_ttl（默认 480 s）
#   bash serve_pd_nixl.sh bidir_proxy          # disagg_proxy_multiturn.py：按 conversation_id 记住 D 的块
#   bash serve_pd_nixl.sh lease [S]            # P 的 kv_lease_duration 改成 S 秒（默认 10），配合 kill D 看释放
#   bash serve_pd_nixl.sh bench [RATE] [N]     # 8k 输入 / 256 输出 random 数据集，打到 proxy（默认 0.4 req/s，100 条）
#   bash serve_pd_nixl.sh sweep                # RATE 0.2 0.4 0.6 1.0 2.0 依次跑 bench，对应 sim_pd.py interference
#   bash serve_pd_nixl.sh multiturn            # benchmark_serving_multi_turn.py --send-conversation-id
#   bash serve_pd_nixl.sh sgl_p | sgl_d | sgl_router   # SGLang 对照：NIXL 后端，bootstrap 8998，router 8000
#
# 环境变量：MODEL（默认 Qwen/Qwen2.5-7B-Instruct，与 vLLM 拆分指南同款）、OUT、VLLM_SRC（vLLM 源码目录，proxy 脚本在里面）、
#           GMU、MAXLEN、UCX_TLS（单机默认 cuda_ipc,cuda_copy,tcp）。
# 日志里要找的几行（vllm 0d6f92a / sglang 8fabf5e）：
#   "Initializing NIXL worker"                    v1/nixl/base_worker.py:631（打印本实例 engine_id）
#   "Registering KV_Caches. use_mla"              base_worker.py:1565（kv_buffer_device、是否走 host buffer）
#   "Detected kv cache layout"                    base_worker.py:1009（异构 TP 要求 HND）
#   "Transfer plan:"                              base_worker.py:2557（D 首次握手后打印 P→D 的 rank 映射）
#   "KV Transfer metrics:"                        v1/metrics.py:108（每 10 秒：Num successful transfers、Avg/P90 xfer time、
#                                                 Avg MB per transfer、Throughput、Avg number of descriptors、Num KV expired reqs）
#   "Releasing expired KV blocks"                 base_worker.py:3402（P 侧租约到期）
#   "Potentially invalid KV blocks"               pull_worker.py:676（D 的完成通知晚于到期）
#   "nixl-push-writer thread started"             push_worker.py:153（push 模式 P 侧后台写线程）
#   /metrics: vllm:nixl_xfer_time_seconds、vllm:nixl_post_time_seconds、vllm:nixl_bytes_transferred、
#             vllm:nixl_num_descriptors、vllm:nixl_num_failed_transfers、vllm:nixl_num_kv_expired_reqs、
#             vllm:nixl_num_notifications_after_expiry
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
OUT=${OUT:-results}
VLLM_SRC=${VLLM_SRC:-$HOME/vllm}
GMU=${GMU:-0.90}
MAXLEN=${MAXLEN:-16384}
export UCX_TLS=${UCX_TLS:-cuda_ipc,cuda_copy,tcp}   # 单机：卡间走 cuda_ipc；跨机换成 rc 或 all
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
PATTERN='NIXL worker|Registering KV_Caches|kv cache layout|Transfer plan|KV Transfer metrics|expired|Potentially invalid|push-writer|PUSH_REG|KV cache size'
PUSH_ENGINE=prefill-engine-001

mkdir -p "$OUT"

serve() {  # serve <tag> <gpu> <port> <side_channel_port> [vllm serve args...]
  local tag=$1 gpu=$2 port=$3 sc=$4; shift 4
  echo "== GPU$gpu :$port side-channel :$sc  vllm serve $MODEL $*"
  CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=$sc \
    vllm serve "$MODEL" --port "$port" --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered "$PATTERN" || true
}

KTC() {  # KTC <connector> <role> [extra_config 片段] [顶层片段]
  local extra=${3:-}
  echo "{\"kv_connector\":\"$1\",\"kv_role\":\"$2\",\"kv_load_failure_policy\":\"fail\"${4:-}\
,\"kv_connector_extra_config\":{${extra}}}"
}

PROXIES=$VLLM_SRC/examples/disaggregated/disaggregated_serving

bench() {  # bench <tag> <port> <rate> <n>
  vllm bench serve --model "$MODEL" --port "$2" --dataset-name random \
    --random-input-len 8192 --random-output-len 256 --ignore-eos \
    --request-rate "$3" --num-prompts "$4" --seed 1 --percentile-metrics ttft,itl,tpot --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" 2>&1 | tee -a "$OUT/bench_$1.log" | \
    grep -E "Median TTFT|P99 TTFT|Median ITL|P99 ITL|Request throughput" || true
}

case "${1:-}" in
  colo)       # 两个独立实例；bench colo 用 vllm-router 或任意轮询 LB 打到 8100/8200
              serve colo_p 0 8100 5600 & serve colo_d 1 8200 5601; wait ;;
  p)          serve pull_p 0 8100 5600 --kv-transfer-config "$(KTC NixlConnector kv_producer)" ;;
  d)          serve pull_d 1 8200 5601 --kv-transfer-config "$(KTC NixlConnector kv_consumer)" ;;
  proxy)      python3 "$VLLM_SRC/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" --port 8192 \
                --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 ;;
  push_p)     serve push_p 0 8100 5600 --kv-transfer-config \
                "$(KTC NixlPushConnector kv_producer '' ",\"engine_id\":\"$PUSH_ENGINE\"")" ;;
  push_d)     serve push_d 1 8200 5601 --no-enable-prefix-caching --kv-transfer-config \
                "$(KTC NixlPushConnector kv_consumer)" ;;
  push_proxy) python3 "$PROXIES/disagg_proxy_pushconnector_demo.py" --model "$MODEL" --port 8192 \
                --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200 \
                --prefill-engine-id "$PUSH_ENGINE" --prefill-kv-host 127.0.0.1 \
                --prefill-side-channel-port 5600 --prefill-tp-size 1 --prefill-pp-size 1 ;;
  bidir_p)    serve bidir_p 0 8100 5600 --no-enable-prefix-caching --kv-transfer-config \
                "$(KTC NixlConnector kv_producer '"bidirectional_kv_xfer":true,"kv_recompute_threshold":64')" ;;
  bidir_d)    serve bidir_d 1 8200 5601 --kv-transfer-config \
                "$(KTC NixlConnector kv_consumer '"bidirectional_kv_xfer":true,"decoder_kv_blocks_ttl":480')" ;;
  bidir_proxy) python3 "$PROXIES/disagg_proxy_multiturn.py" --port 8192 \
                --prefiller-host 127.0.0.1 --prefiller-port 8100 --decoder-host 127.0.0.1 --decoder-port 8200 ;;
  lease)      # 起好 D 和 proxy 后发一批请求，P 算完立刻 kill D：日志里 "Releasing expired KV blocks" 应在 S 秒内出现
              serve lease_p 0 8100 5600 --kv-transfer-config \
                "$(KTC NixlConnector kv_producer "\"kv_lease_duration\":${2:-10}")" ;;
  bench)      bench "r${2:-0.4}" 8192 "${2:-0.4}" "${3:-100}"
              for port in 8100 8200; do
                curl -s "localhost:$port/metrics" | grep -E '^vllm:nixl_' | grep -v '_created' | \
                  tee -a "$OUT/bench_r${2:-0.4}.log" || true
              done ;;
  sweep)      for r in 0.2 0.4 0.6 1.0 2.0; do bench "r$r" 8192 "$r" 100; done ;;
  multiturn)  python3 "$VLLM_SRC/benchmarks/multi_turn/benchmark_serving_multi_turn.py" \
                --model "$MODEL" --url http://127.0.0.1:8192 \
                --input-file "$VLLM_SRC/benchmarks/multi_turn/generate_multi_turn.json" \
                --num-clients 2 --max-active-conversations 6 --send-conversation-id 2>&1 | tee "$OUT/multiturn.log" ;;
  sgl_p)      python -m sglang.launch_server --model-path "$MODEL" --port 30000 --disaggregation-mode prefill \
                --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 2>&1 | tee "$OUT/sgl_p.log" ;;
  sgl_d)      python -m sglang.launch_server --model-path "$MODEL" --port 30001 --base-gpu-id 1 \
                --disaggregation-mode decode --disaggregation-transfer-backend nixl 2>&1 | tee "$OUT/sgl_d.log" ;;
  sgl_router) python -m sglang_router.launch_router --pd-disaggregation \
                --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 127.0.0.1 --port 8192 ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
