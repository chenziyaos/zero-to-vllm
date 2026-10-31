#!/usr/bin/env bash
# VLLM-14 GPU 实验：同一个模型，换 KV 卸载方式起服务，用"前缀总量超过 GPU 池"的负载压测，看外部命中率和 TTFT。
#
# 用法：
#   bash serve_offload.sh base                 # 不开卸载的基线（只有 GPU 前缀缓存）
#   bash serve_offload.sh native [GiB]         # --kv-offloading-backend native，CPU 层默认 128 GiB
#   bash serve_offload.sh arc [GiB]            # 同上，eviction_policy=arc
#   bash serve_offload.sh sync [GiB]           # 同上，sync_load_budget_bytes=2 GiB（短命中当步回迁）
#   bash serve_offload.sh small                # CPU 层 32 GiB < GPU 池：验证"只是镜像，命中率不涨"
#   bash serve_offload.sh simple [GiB]         # VLLM_USE_SIMPLE_KV_OFFLOAD=1 → SimpleCPUOffloadConnector
#   bash serve_offload.sh tiered [DIR]         # TieringOffloadingSpec：CPU 64 GiB + fs 层（默认 /mnt/kv_cache）
#   bash serve_offload.sh canonical [DIR]      # TP=4 + canonical_layout + fs 层（与其他 TP 的实例共享）
#   bash serve_offload.sh lmcache [GiB]        # LMCache MP 模式：先另开终端 `lmcache server --port 5555 --l1-size-gb GiB`
#   bash serve_offload.sh mooncake             # MooncakeStoreConnector 单机（需要 mooncake_master 与 MOONCAKE_CONFIG_PATH）
#   bash serve_offload.sh bench [N] [C]        # prefix_repetition：N 个 16K 前缀（默认 64）× 10 轮，C 并发（默认 16）
#   bash serve_offload.sh sgl_base|sgl_hicache|sgl_hicache_big|sgl_selective|sgl_file   # SGLang 对照
#
# 环境变量：MODEL（默认 meta-llama/Llama-3.1-8B-Instruct）、OUT、PORT、GMU。
# 日志里要找的几行（vllm 0d6f92a / sglang 8fabf5e）：
#   "Creating offloading spec with name"          v1/kv_offload/factory.py:57
#   "Allocating %d CPU tensors"                   v1/kv_offload/cpu/gpu_worker.py:881
#   "External prefix cache hit rate"              v1/metrics/loggers.py:329（每 10 秒一行，连接器口径的命中率）
#   /metrics: vllm:external_prefix_cache_queries / _hits、vllm:kv_offload_load_bytes / _load_time / _store_bytes、
#             vllm:kv_offload_lookup_sync_delay_seconds、vllm:kv_offload_config_info
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
OUT=${OUT:-results}
PORT=${PORT:-8000}
GMU=${GMU:-0.92}
PATTERN='offloading spec|CPU tensors|External prefix cache hit rate|KV cache size|Maximum concurrency|offload|LMCache|Mooncake'

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
    tee "$OUT/$tag.log" | grep -E --line-buffered 'HiCache|hicache|host memory|#cached-token|KV Cache' || true
}

KTC() {  # $1 = kv_connector_extra_config 的 JSON 片段
  echo "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$1}}"
}

case "${1:-}" in
  base)      serve base ;;
  native)    serve "native_${2:-128}" --kv-offloading-backend native --kv-offloading-size "${2:-128}" ;;
  small)     serve native_32 --kv-offloading-backend native --kv-offloading-size 32 ;;
  arc)       serve "arc_${2:-128}" --kv-transfer-config \
               "$(KTC "\"cpu_bytes_to_use\":$(( ${2:-128} * 1073741824 )),\"eviction_policy\":\"arc\"")" ;;
  sync)      serve "sync_${2:-128}" --kv-transfer-config \
               "$(KTC "\"cpu_bytes_to_use\":$(( ${2:-128} * 1073741824 )),\"sync_load_budget_bytes\":2147483648")" ;;
  simple)    VLLM_USE_SIMPLE_KV_OFFLOAD=1 serve "simple_${2:-128}" \
               --kv-offloading-backend native --kv-offloading-size "${2:-128}" ;;
  tiered)    serve tiered --kv-transfer-config "$(KTC "\"spec_name\":\"TieringOffloadingSpec\",\
\"cpu_bytes_to_use\":68719476736,\"block_size\":16,\"eviction_policy\":\"lru\",\
\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"${2:-/mnt/kv_cache}\",\"n_read_threads\":32,\"n_write_threads\":16}]")" ;;
  canonical) serve canonical -tp 4 --kv-transfer-config "$(KTC "\"spec_name\":\"TieringOffloadingSpec\",\
\"cpu_bytes_to_use\":64000000000,\"canonical_layout\":true,\
\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"${2:-/mnt/kv}\"}]")" ;;
  lmcache)   serve "lmcache_${2:-128}" --kv-offloading-backend lmcache --kv-offloading-size "${2:-128}" \
               --disable-hybrid-kv-cache-manager ;;
  mooncake)  : "${MOONCAKE_CONFIG_PATH:?先写 mooncake_config.json 并 export MOONCAKE_CONFIG_PATH}"
             serve mooncake --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}' ;;
  bench)
    N=${2:-64}; C=${3:-16}
    vllm bench serve --model "$MODEL" --port "$PORT" --dataset-name prefix_repetition \
      --prefix-repetition-prefix-len 16000 --prefix-repetition-suffix-len 2000 \
      --prefix-repetition-num-prefixes "$N" --prefix-repetition-output-len 128 \
      --num-prompts $((N * 10)) --max-concurrency "$C" --seed 0 \
      --save-result --result-dir "$OUT" 2>&1 | tee -a "$OUT/bench_n${N}_c${C}.log" | \
      grep -E "Mean TTFT|Median TTFT|P99 TTFT|Request throughput|Total token throughput" || true
    curl -s "localhost:$PORT/metrics" | grep -E '^vllm:(external_prefix_cache|prefix_cache|kv_offload_(load|store)_bytes)' \
      | grep -v '_created' | tee -a "$OUT/bench_n${N}_c${C}.log" || true ;;
  sgl_base)        sgl sgl_base ;;
  sgl_hicache)     sgl sgl_hicache --enable-hierarchical-cache ;;                      # ratio 2.0（默认）
  sgl_hicache_big) sgl sgl_hicache_big --enable-hierarchical-cache --hicache-size 137 ;; # 137 GB = 128 GiB，每 rank
  sgl_selective)   sgl sgl_selective --enable-hierarchical-cache --hicache-size 137 \
                     --hicache-write-policy write_through_selective ;;
  sgl_file)        sgl sgl_file --enable-hierarchical-cache --hicache-ratio 1.2 \
                     --hicache-storage-backend file --hicache-storage-prefetch-policy timeout ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
