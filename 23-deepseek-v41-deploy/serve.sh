#!/usr/bin/env bash
# VLLM-23 实战：DeepSeek-V4.1-Flash-1017 在消费级卡和 H20 上起服务。每个档位一行命令，日志落 $OUT/<档位>.log。
#
# 用法：
#   bash serve.sh pro6000x4        # 4×RTX PRO 6000 96G，TP4 EP，Engram 放主机，1M 上下文（消费级首选）
#   bash serve.sh pro6000x8        # 8×RTX PRO 6000，TP8 EP，Engram 留显存（消费级高配）
#   bash serve.sh pro6000x1        # 1×RTX PRO 6000 + 768 GB 内存，UVA 卸 214 GiB 专家，128K（入门）
#   bash serve.sh rtx5090x8        # 8×RTX 5090 32G，每卡卸 9 GiB 专家，128K
#   bash serve.sh rtx4090_48gx8    # 8×4090 48G 改装卡，只能走 SM89 分叉镜像（FORK_IMAGE）
#   bash serve.sh h20x8            # 8×H20 96G，套 recipe 的 H100 模板（Engram 放主机，batched 4096）
#   bash serve.sh h20x8_gpu        # 8×H20 96G，Engram 留显存，上下文降到 256K，不吃主机内存
#   bash serve.sh h20e_tp8         # 8×H20-3e 141G，TP8，Engram 留显存（数据中心首选）
#   bash serve.sh h20e_tp4dp2      # 8×H20-3e 141G，TP4×DP2，Engram 放主机共享 /dev/shm（高吞吐）
#   bash serve.sh sgl_h20e         # SGLang 对照（8×H20-3e TP8 EP8，需另装 sglang）
#   bash serve.sh plan <档位>      # 不起服务，只打印 deploy_planner.py 对这一档的显存账
#
# 环境变量：MODEL（默认 deepseek-ai/DeepSeek-V4.1-Flash-1017）、OUT、PORT、GMU（覆盖档位默认值）、
#   ADAPTIVE（默认 true；SM120 起不来时设 false）、SPEC（DSpark 草稿数，默认 5）、TEXT_ONLY=1（加 --language-model-only）、
#   FORK_IMAGE（SM89 分叉镜像）、HF_HOME。
#
# 日志里要找的几行（行号对应 vllm 0d6f92a）：
#   "Using '...' Mxfp4 MoE backend."                     model_executor/layers/fused_moe/oracle/mxfp4.py:556
#   "Running checkpoint MXFP8 MoE activations as block FP8"  oracle/mxfp4.py:544（Hopper，H20 会打）
#   "Engram table offloaded to pinned host memory"       models/deepseek_v41/common/engram.py:1067
#   "Engram node-local DP group size: %d (TP=%d)"        distributed/parallel_state.py:2245（DP>1 共享 /dev/shm）
#   "Total CPU offloaded parameters"                     model_executor/offloader/uva.py:69（--cpu-offload-gb 生效）
#   "Available KV cache memory: X GiB"                   v1/worker/gpu_worker.py:776
#   "GPU KV cache size: N tokens, Maximum concurrency"   v1/core/kv_cache_utils.py:1148
#   "requires a FlashInfer DSV4 sparse MLA decode specialization"  nvidia/flashinfer_sparse.py:656（SM120 缺 #4380）
#   "Adaptive verification could not profile step costs" v1/worker/gpu/spec_decode/adaptive_verification.py:360
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
OUT=${OUT:-results}
PORT=${PORT:-8000}
ADAPTIVE=${ADAPTIVE:-true}
SPEC=${SPEC:-5}
FORK_IMAGE=${FORK_IMAGE:-ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130}
HERE=$(cd "$(dirname "$0")" && pwd)
PATTERN='Mxfp4 MoE backend|MXFP8 MoE activations|Engram table offloaded|Engram node-local|CPU offloaded|Available KV cache memory|KV cache size|Maximum concurrency|#4380|Adaptive verification|Error|error:|out of memory'

# recipe 的 base_env：首次加载要从磁盘读 475 GiB，默认 600 s 不够
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT"

spec_cfg() {
  printf '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":%s}' \
    "$SPEC" "$ADAPTIVE"
}

# recipe 的 base_args + features（tokenizer / reasoning / tool parser + DSpark）
COMMON=(--tokenizer-mode deepseek_v41
        --reasoning-parser deepseek_v41
        --enable-auto-tool-choice --tool-call-parser deepseek_v41
        --enable-prefix-caching)
[[ "${TEXT_ONLY:-0}" == 1 ]] && COMMON+=(--language-model-only)

serve() {
  local tag=$1 gmu=$2; shift 2
  echo "== vllm serve $MODEL --gpu-memory-utilization ${GMU:-$gmu} $*"
  vllm serve "$MODEL" --port "$PORT" --gpu-memory-utilization "${GMU:-$gmu}" \
    "${COMMON[@]}" --speculative-config "$(spec_cfg)" "$@" 2>&1 | \
    tee "$OUT/$tag.log" | grep -E --line-buffered "$PATTERN" || true
}

sm120_env() {
  # SM120（RTX PRO 6000 / 5090）主线走 FLASHINFER_MLA_SPARSE_DSV41，FlashInfer 需含 flashinfer-ai/flashinfer#4380。
  # 先查一遍 dispatch 表（vllm/utils/flashinfer.py:505 查的就是它），缺了在加载权重前就报，
  # 省得等 5 分钟才看到 RuntimeError。表里要有 (padded_heads, SWA 宽度) 这一项，宽度随 DSpark 草稿数变。
  python - <<'EOF' || { echo "!! FlashInfer 缺 SM120 DSV4 sparse MLA 特化（flashinfer#4380），见 README 排障"; exit 1; }
import importlib, sys
try:
    mod = importlib.import_module("flashinfer.mla._sparse_mla_sm120")
except ImportError:
    sys.exit(1)
table = getattr(mod, "_DECODE_DSV4_DISPATCH", None)
if not table:
    sys.exit(1)
print("FlashInfer SM120 DSV4 decode 特化 (heads, top_k):", sorted(table))
EOF
}

case "${1:-}" in
  pro6000x4)
    sm120_env
    serve pro6000x4 0.93 --tensor-parallel-size 4 --enable-expert-parallel \
      --kv-cache-dtype fp8 --max-model-len 1048576 --max-num-seqs 16 --max-num-batched-tokens 4096 \
      --engram-config '{"cpu_offload":true}' --safetensors-load-strategy lazy ;;
  pro6000x8)
    sm120_env
    serve pro6000x8 0.92 --tensor-parallel-size 8 --enable-expert-parallel \
      --kv-cache-dtype fp8 --max-model-len 1048576 --max-num-seqs 64 --max-num-batched-tokens 8192 \
      --engram-config '{"cpu_offload":false}' ;;
  pro6000x1)
    sm120_env
    serve pro6000x1 0.92 --kv-cache-dtype fp8 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 4096 \
      --engram-config '{"cpu_offload":true}' \
      --offload-backend uva --cpu-offload-gb 214 \
      --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight ;;
  rtx5090x8)
    sm120_env
    serve rtx5090x8 0.95 --tensor-parallel-size 8 --enable-expert-parallel \
      --kv-cache-dtype fp8 --max-model-len 131072 --max-num-seqs 32 --max-num-batched-tokens 8192 \
      --engram-config '{"cpu_offload":true}' \
      --offload-backend uva --cpu-offload-gb 9 \
      --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight ;;
  rtx4090_48gx8)
    # 主线没有 SM89 的 sparse MLA 后端；下面是 yhfgyyf 分叉 README 的 8×4090 48G 命令，ced_prefill 只在分叉里有
    echo "== docker run $FORK_IMAGE（SM89 分叉，非主线）"
    docker run --rm --gpus all --ipc host --shm-size 32g -p "$PORT:8000" \
      -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
      -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
      "$FORK_IMAGE" vllm serve "$MODEL" --tensor-parallel-size 8 --distributed-executor-backend mp \
      --enable-expert-parallel --kv-cache-dtype fp8 --block-size 128 --max-model-len auto \
      --max-num-seqs 4 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.98 \
      --engram-config '{"cpu_offload":true}' --safetensors-load-strategy lazy \
      "${COMMON[@]}" --hf-overrides '{"ced_prefill":true}' --speculative-config "$(spec_cfg)" 2>&1 | \
      tee "$OUT/rtx4090_48gx8.log" | grep -E --line-buffered "$PATTERN" || true ;;
  h20x8)
    # recipe 的 H100 覆盖原样搬过来：H20 96G 比 H100 多 16 GB，KV 池更宽
    serve h20x8 0.92 --tensor-parallel-size 8 --enable-expert-parallel \
      --max-model-len 1048576 --max-num-seqs 64 --max-num-batched-tokens 4096 \
      --engram-config '{"cpu_offload":true}' ;;
  h20x8_gpu)
    serve h20x8_gpu 0.92 --tensor-parallel-size 8 --enable-expert-parallel \
      --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 4096 \
      --engram-config '{"cpu_offload":false}' ;;
  h20e_tp8)
    serve h20e_tp8 0.92 --tensor-parallel-size 8 --enable-expert-parallel \
      --max-model-len 1048576 --max-num-seqs 128 --max-num-batched-tokens 8192 \
      --engram-config '{"cpu_offload":false}' ;;
  h20e_tp4dp2)
    # H200 模板：TP4 每副本，DP2 两个副本；cpu_offload 且 DP>1 时默认 dp_shared_memory，整机一份表
    serve h20e_tp4dp2 0.92 --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel \
      --max-model-len 1048576 --max-num-seqs 64 --max-num-batched-tokens 8192 \
      --engram-config '{"cpu_offload":true}' ;;
  sgl_h20e)
    echo "== sglang.launch_server --model-path $MODEL --tp-size 8 --ep-size 8"
    python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --tp-size 8 --ep-size 8 \
      --trust-remote-code --mem-fraction-static "${GMU:-0.90}" 2>&1 | \
      tee "$OUT/sgl_h20e.log" | grep -E --line-buffered 'KV Cache|max_total_num_tokens|backend|Error' || true ;;
  plan)
    P="python3 $HERE/deploy_planner.py --scenario mem"
    case "${2:-pro6000x4}" in
      pro6000x4)   $P --gpu rtxpro6000 --n 4 --tp 4 --gmu 0.93 --max-num-seqs 16 --batched 4096 ;;
      pro6000x8)   $P --gpu rtxpro6000 --n 8 --tp 8 --engram gpu --gmu 0.92 --max-num-seqs 64 --batched 8192 ;;
      pro6000x1)   $P --gpu rtxpro6000 --n 1 --tp 1 --offload-gib 214 --gmu 0.92 --max-model-len 131072 --max-num-seqs 2 ;;
      rtx5090x8)   $P --gpu rtx5090 --n 8 --tp 8 --offload-gib 9 --gmu 0.95 --max-model-len 131072 --max-num-seqs 32 --batched 8192 ;;
      rtx4090_48gx8) $P --gpu rtx4090_48g --n 8 --tp 8 --gmu 0.98 --max-model-len 262144 --max-num-seqs 4 ;;
      h20x8)       $P --gpu h20 --n 8 --tp 8 --gmu 0.92 --max-num-seqs 64 ;;
      h20x8_gpu)   $P --gpu h20 --n 8 --tp 8 --engram gpu --gmu 0.92 --max-model-len 262144 --max-num-seqs 64 ;;
      h20e_tp8)    $P --gpu h20_141g --n 8 --tp 8 --engram gpu --gmu 0.92 --max-num-seqs 128 --batched 8192 ;;
      h20e_tp4dp2) $P --gpu h20_141g --n 8 --tp 4 --gmu 0.92 --max-num-seqs 64 --batched 8192 ;;
      *) echo "未知档位 $2"; exit 1 ;;
    esac ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
