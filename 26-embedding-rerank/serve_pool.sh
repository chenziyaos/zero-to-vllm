#!/usr/bin/env bash
# VLLM-26 GPU 实验：embedding 和 rerank 服务。单卡即可，0.6B 模型 24 GB 显存足够。本脚本没有在 GPU 上实跑过。
#
# 用法（每个 serve 模式起一个前台服务，另开终端跑 bench / client）：
#   bash serve_pool.sh emb [MODEL]       # Qwen3-Embedding（默认 0.6B），带 is_matryoshka 覆盖，允许请求里带 dimensions
#   bash serve_pool.sh m3                # BAAI/bge-m3：双向注意力，chunked prefill 和 prefix caching 默认关
#   bash serve_pool.sh rr_bge            # BAAI/bge-reranker-v2-m3：cross-encoder
#   bash serve_pool.sh rr_qwen           # Qwen/Qwen3-Reranker-0.6B：按官方说明改成 Qwen3ForSequenceClassification
#   bash serve_pool.sh bench_emb [LEN]   # vllm bench serve 打 /v1/embeddings，默认每条 512 token
#   bash serve_pool.sh bench_rr [DOCS]   # vllm bench serve 打 /rerank，random-rerank 数据集，默认每次 100 篇
#   bash serve_pool.sh client            # pool_client.py：embed、search（bi-encoder 召回）、rerank（cross-encoder 精排）
#
# 环境变量：PORT（默认 8000）、OUT（结果目录，默认 results）、VLLM_SRC（vLLM 源码目录，rr_qwen 要用里面的 jinja 模板）
#
# 日志里要找的行：
#   "This model does not officially support chunked prefill / prefix caching"   手动打开了不该开的开关
#   "Disabling chunked prefill: model has non-causal attention layers."         双向模型被引擎强制关掉
#   /metrics: vllm:prefix_cache_queries / vllm:prefix_cache_hits                 rr_qwen 的共享前缀到底命中了多少
set -Eeuo pipefail

MODE=${1:-}
PORT=${PORT:-8000}
OUT=${OUT:-results}
VLLM_SRC=${VLLM_SRC:-$HOME/vllm}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)

case "$MODE" in
  emb)
    MODEL=${2:-Qwen/Qwen3-Embedding-0.6B}
    vllm serve "$MODEL" --runner pooling --port "$PORT" \
      --hf-overrides '{"is_matryoshka": true}' 2>&1 | tee "$OUT/emb.log" ;;
  m3)
    vllm serve BAAI/bge-m3 --port "$PORT" 2>&1 | tee "$OUT/m3.log" ;;
  rr_bge)
    vllm serve BAAI/bge-reranker-v2-m3 --port "$PORT" 2>&1 | tee "$OUT/rr_bge.log" ;;
  rr_qwen)
    vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling --port "$PORT" \
      --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
      --chat-template "$VLLM_SRC/examples/pooling/score/template/qwen3_reranker.jinja" 2>&1 | tee "$OUT/rr_qwen.log" ;;
  bench_emb)
    LEN=${2:-512}
    MODEL=$(curl -s "localhost:$PORT/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
    for RATE in 50 200 800 inf; do
      vllm bench serve --backend openai-embeddings --endpoint /v1/embeddings --model "$MODEL" --port "$PORT" \
        --dataset-name random --random-input-len "$LEN" --num-prompts 2000 --request-rate "$RATE" \
        --save-result --result-dir "$OUT" --result-filename "bench_emb_${LEN}_${RATE}.json"
    done ;;
  bench_rr)
    DOCS=${2:-100}
    MODEL=$(curl -s "localhost:$PORT/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
    vllm bench serve --backend vllm-rerank --endpoint /rerank --model "$MODEL" --port "$PORT" \
      --dataset-name random-rerank --random-input-len 512 --random-batch-size "$DOCS" --num-prompts 2000 \
      --save-result --result-dir "$OUT" --result-filename "bench_rr_$DOCS.json"
    curl -s "localhost:$PORT/metrics" | grep -E '^vllm:prefix_cache_(queries|hits)' || true ;;
  client)
    MODEL=$(curl -s "localhost:$PORT/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
    case "$MODEL" in
      *Reranker*|*reranker*)
        python3 "$HERE/pool_client.py" rerank --url "http://localhost:$PORT" --model "$MODEL" --engine vllm \
          --query "显存不够怎么办" ;;
      *)
        python3 "$HERE/pool_client.py" embed --url "http://localhost:$PORT" --model "$MODEL"
        python3 "$HERE/pool_client.py" embed --url "http://localhost:$PORT" --model "$MODEL" --dimensions 256
        python3 "$HERE/pool_client.py" search --url "http://localhost:$PORT" --model "$MODEL" --query "显存不够怎么办"
        python3 "$HERE/pool_client.py" bench --url "http://localhost:$PORT" --model "$MODEL" ;;
    esac ;;
  *)
    sed -n '2,18p' "$0"; exit 1 ;;
esac
