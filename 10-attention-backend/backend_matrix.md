# 注意力后端：选择过程与能力表

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。表里的“行”都是这两个提交的行号。

## 1. 选择链路（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| v1/attention/selector.py 134–138 | `get_attn_backend` | 只有用户显式给了 `--block-size`，才把 block_size 传给选择器；否则传 None，block 不参与筛选 |
| platforms/cuda.py 84–181 | `_get_backend_priorities` | 按 (是否 MLA, SM 架构, 是否 causal, 是否 mm_prefix, KV dtype, 头数) 给出候选顺序 |
| platforms/cuda.py 393–421 | `get_valid_backends` | 逐个调 `validate_configuration`；ImportError / OSError 也转成原因字符串 `"类型: 信息"` |
| platforms/cuda.py 424–452 | 用户指定分支 | 只验这一个；不合法时抛 `ValueError("Selected backend X is not valid for this configuration. Reason: [...]")`（447），合法时打 `"Using %s backend."`（452） |
| platforms/cuda.py 470–480 | 全部不合法 | `"No valid attention backend found for … Reasons: {…}"` |
| platforms/cuda.py 495–510 | block 挤占告警 | 用户给了 `--block-size`，且排在前面的后端**唯一**原因是 `block_size not supported` 时告警 |
| platforms/cuda.py 513 | 成功日志 | `"Using %s attention backend out of potential backends: [...]"` |
| platforms/interface.py 755–785 | `update_block_size_for_backend` | 用户没给 block 时取 `get_preferred_block_size(16)`；不等于 16 时打 `"Setting kv cache block size to %d for %s backend(s)."`（782） |
| v1/attention/backend.py 127–143 | `supports_block_size` | block 是任一支持尺寸（或 MultipleOf 基数）的倍数即可：框架 block 与内核 block 解耦 |
| v1/attention/backend.py 160–167 | `get_preferred_block_size` | 默认 16 能用就 16，否则取支持列表的最小值 |
| v1/attention/backend.py 284–373 | `validate_configuration` | 依次检查：head_size、dtype、kv_cache_dtype、block_size、mm_prefix、MLA、sinks、sparse、per-head 量化 scale、compute capability、attn_type、sliding window、R-SWA、non-causal、MLA+DCP 非 causal、batch invariance、KV connector、PCP、DCP、自适应验证、`supports_combination` |
| v1/attention/backend.py 570 | `AttentionCGSupport` | ALWAYS 3 / UNIFORM_BATCH 2 / UNIFORM_SINGLE_TOKEN_DECODE 1 / NEVER 0，交给 `resolve_cudagraph_mode_and_sizes` 降级（见 09-model-runner） |

## 2. 候选顺序（platforms/cuda.py 84–181）

| 情形 | 顺序 |
|---|---|
| 稠密，SM10x，causal | FLASHINFER → FLASH_ATTN → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT（mm_prefix 时最前加 TRITON_FLASHINFER） |
| 稠密，其他 | FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT（SM90 + mm_prefix 时最前加 TRITON_FLASH_ATTN） |
| MLA，SM10x | FLASHINFER_MLA → TOKENSPEED_MLA → CUTLASS_MLA → FLASH_ATTN_MLA → FLASHMLA → TRITON_MLA → 稀疏表 |
| MLA，SM12x | TRITON_MLA → FLASHINFER_MLA_SPARSE_SM120 |
| MLA，其他 | FLASH_ATTN_MLA → FLASHMLA → FLASHINFER_MLA → TRITON_MLA → 稀疏尾 |

SM10x 稀疏表随 KV dtype 和头数变：fp8 → [FLASHINFER_MLA_SPARSE, FLASHMLA_SPARSE]；BF16 且头数 ≤ 16 → [FLASH_ATTN_MLA_SPARSE_FA4, FLASHINFER_MLA_SPARSE, FLASHMLA_SPARSE]；其余 → [FLASHMLA_SPARSE, FLASHINFER_MLA_SPARSE]。其他架构的稀疏尾是 FLASH_ATTN_MLA_SPARSE、FLASHMLA_SPARSE，head_size 512 时 FLASHINFER_MLA_SPARSE_SM90 排最前，否则排最后。TOKENSPEED_MLA 的注释写明“R1 维度 + FP8 KV 才用，bs≈8 以上占优，bs≤2 退化”。

## 3. 稠密后端能力

| 后端 | 文件 / 行 | KV dtype | head_size | block | CC | sinks | CG 支持 |
|---|---|---|---|---|---|---|---|
| FLASH_ATTN | flash_attn.py 286–500, 619 | auto/fp16/bf16/fp8/fp8_e4m3（FP8 要 FA3@SM90 或 FA4@SM100） | %8，≤256（FA4 可到 512） | MultipleOf(16)；FA4 hd256 → [128]；SM90 FP8 hd512 FA4 → [64] | ≥ 8.0 | FA3/FA4 且 CC ≥ 9.0 | FA3/4 ALWAYS，FA2 UNIFORM_BATCH |
| FLASHINFER | flashinfer.py 572–730, 1176–1225 | 再加 fp8_e5m2、nvfp4（条件） | 64/128/256/512 | [16,32,64]；SM100 + trtllm + GQA 时到 1024 | 8.0–12.1 | SM90/12x 要 XQA decode，SM100 要 trtllm 两段 | UNIFORM_BATCH（trtllm/XQA 可用或 SM8x FA2 decode），否则 USTD |
| TRITON_ATTN | triton_attn.py 97, 319–405 | 最全：含 fp8_e5m2、int4/int8/fp8 per-token-head | ≥ 32 | MultipleOf(16) | 任意 | 支持 | ALWAYS |
| FLEX_ATTENTION | flex_attention.py 88–160, 836 | auto/fp16/bf16 | 任意 | MultipleOf(16) | 任意 | 不支持 | ALWAYS |
| TURBOQUANT | turboquant_attn.py | 仅 turboquant_* | — | — | — | — | USTD |

FA 版本（fa_utils.py 94–107）：SM9x → 3，SM10x → 4，其余 → 2；`flash_attn_supports_kv_cache_dtype`（282–313）对 e5m2 永远 False。FlashInfer 的 trtllm 判定在 utils/flashinfer.py 719–765：batch invariant 或拿不到 cubin 时直接 False；SM90/SM12x 只有 decode（XQA），SM100 两段都有。

## 4. MLA 后端能力

| 后端 | 文件 / 行 | block | CC | KV dtype | CG 支持 | supports_combination |
|---|---|---|---|---|---|---|
| FLASHINFER_MLA | mla/flashinfer_mla.py 140, 204–240 | [32, 64] | 10.x | MLA_KV | ALWAYS | qk_nope ∈ {64,128,192} |
| TOKENSPEED_MLA | mla/tokenspeed_mla.py 77 | [32, 64] | 10.x | fp8/fp8_e4m3 | UNIFORM_BATCH | 要装包、要 R1 维度 128/64/128 |
| CUTLASS_MLA | mla/cutlass_mla.py 33, 49–65 | [128] | 10.x | MLA_KV | USTD | — |
| FLASH_ATTN_MLA | mla/flashattn_mla.py 49–90, 127 | MultipleOf(16) | 9.x | auto/fp16/bf16 | UNIFORM_BATCH | FA3 + SM90；head 576 |
| FLASHMLA | mla/flashmla.py 57–90, 110 | [64] | 9.x/10.x | MLA_KV | UNIFORM_BATCH | 稠密只在 Hopper |
| TRITON_MLA | mla/triton_mla.py 77 | MultipleOf(16) | 任意 | MLA_KV | UNIFORM_BATCH | — |

MLA_KV = auto/fp16/bf16/fp8/fp8_e4m3。MLA 层传给选择器的 head_size = kv_lora_rank + qk_rope_head_dim（DeepSeek-V3 为 576）。计算路径在 model_executor/layers/attention/mla_attention.py：decode token 走 `forward_mqa`（吸收 W_UK，1088–1105 用 `W_UK_T` 做 bmm），prefill token 走 `forward_mha`（解压出每头 K/V），`process_weights_after_loading`（1242–1354）把 W_UK 转置存成 `W_UK_T`。

## 5. 元数据构建：FA 与 FlashInfer 的差别

| 位置 | 说明 |
|---|---|
| flash_attn.py 707–708 | `aot_schedule = (FA 版本 == 3)`：只有 FA3 在 build 时预算调度元数据 |
| flash_attn.py 752–771 | 开 FULL CUDA graph 且 aot_schedule 时预分配 `scheduler_metadata`，`max_num_splits` 取 `flash_attn_max_num_splits_for_cuda_graph` |
| flash_attn.py 832–837 | 投机解码的 proposer（fast_build）和 batch invariant 关掉 AOT 调度 |
| flashinfer.py 1487–1497 | decode wrapper `BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)` |
| flashinfer.py 1975–1985 | decode 用 `fast_plan_decode`：plan 在 CPU 上算 indptr / last_page_len，pin 住再异步拷贝 |
| flashinfer.py 2895–2910 | 第一次调用或非 CUDA graph 时跑原始 `plan()`，之后复用 `_cached_module` |
| flashinfer.py 20–21, 403 | trtllm 路径直接调 `trtllm_batch_decode_with_kv_cache` / `trtllm_batch_context_with_kv_cache`，不经过 wrapper.plan |

## 6. SGLang 对照

| 位置 | 说明 |
|---|---|
| srt/arg_groups/model_override_base.py 277–351 | `get_default_attn_backend`：MHA 在 Hopper（CUDA ≥ 12.3）且不开投机或 topk=1 → fa3（issue #17411：flashinfer 0.6.1 在 Hopper 退化）；SM100 → trtllm_mha（非对称 KV → fa4）；HIP → aiter；否则 flashinfer（模型有 sinks → triton）。MLA：Hopper → fa3，SM100 → flashinfer，HIP 头数 16/128 → aiter，其余 triton |
| srt/arg_groups/model_overrides/deepseek_v2.py 179–188 | DeepSeek V3/R1 在 SM100 且三个后端参数都没给 → trtllm_mla |
| srt/arg_groups/model_overrides/deepseek_v2.py 99–104 | DSA 模型（V3.2 / GLM-5）没给后端 → dsa |
| srt/models/deepseek_common/attention_backend_handler.py 108–141 | 每个 batch 选计算路径：在 piecewise / breakable CUDA graph 里或开了 CP → MLA；无投机的 extend 且（前缀 ≥ chunked_prefix_cache_threshold 或没有前缀）→ MHA_ONE_SHOT（总长 ≤ chunk 容量）或 MHA_CHUNKED_KV；其余 → MLA |
| srt/layers/attention/hybrid_attn_backend.py | `--prefill-attention-backend` / `--decode-attention-backend` 分别指定，按 forward mode 分派 |
| srt/layers/attention/attention_registry.py | 后端注册表，`--attention-backend` 的取值来源 |

## 7. 读日志的顺序

1. `Using X attention backend out of potential backends: [...]`：括号里是全部合法候选，按优先级排列。
2. `Setting kv cache block size to N`：没给 `--block-size` 时，后端把 block 改成了自己的首选。
3. `--block-size N precluded higher-priority backend(s)`：你给的 block 把更快的后端挤掉了。
4. `cudagraph_mode` 相关的降级：后端的 CG 支持级别决定 FULL 能不能保住。
