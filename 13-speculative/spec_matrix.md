# 投机解码：源码位置与参数对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。表里的“行”都是这两个提交的行号。vllm 路径相对 `vllm/`，sglang 路径相对 `python/sglang/`。

## 1. 配置（vllm `config/speculative.py`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `method` | 推断 | `eagle` / `eagle3` / `mtp` / `ngram` / `ngram_gpu` / `suffix` / `draft_model` / `dflash` / `dspark` …… |
| `num_speculative_tokens` | 必填 | k，草稿长度 |
| `prompt_lookup_min` / `prompt_lookup_max` | 534；默认值 1384–1400 | 都不写 = 5/5；只写 max 时 min = max（docstring 说默认 1，代码不是） |
| `suffix_decoding_max_tree_depth` | 24（566） | SuffixDecoding 的后缀树深度 |
| `num_speculative_tokens_per_batch_size` | None（551） | 动态 k：`[[start_bs, end_bs, k], ...]` |
| `rejection_sample_method` | `standard`（590） | `standard` / `block`（只 MRV2）/ `synthetic` / `fly` |
| `synthetic_acceptance_length` / `_rates` | 604 / 610 | 只在 synthetic 下合法，二选一 |
| `enable_adaptive_verification` | False（613） | 每步按存活概率分配验证预算；要求 full cudagraph，不兼容 LoRA、PP |
| `draft_sample_method` | `greedy`（662） | `probabilistic` 时缓存草稿 logits，接受率 = Σmin(p,q) |
| `fly_window_size` / `fly_entropy_threshold` / `fly_entropy_top_k` | min(6,k-1) / 0.3 / 3（680–687） | FLy；k ≥ 2；启动打 lossy WARNING（2115） |

## 2. 调度与 KV（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| v1/core/sched/scheduler.py 372 | `_spec_decode_step_blocks` | 每请求每步预留 `cdiv(1 + k + lookahead, block_size)` 个块 |
| 同上 2354 | 回退 | `num_computed_tokens -= num_rejected`；被拒位置的 KV 槽下一步直接覆盖 |
| config/vllm.py `num_lookahead_tokens` | | DFlash k+1；EAGLE 家族、draft_model k；ngram / suffix 0 |
| v1/spec_decode/dynamic/utils.py 7 / 77 | 校验 / 稠密表 | 第一档从 1 开始、不重叠、k ≥ 0；空档和尾部沿用前一档；`min(k_i, k)` 截顶 |
| config/vllm.py 1084–1100 | `_maybe_override_dynamic_sd_cudagraph_mode` | MRV1 + 动态 k：FULL → PIECEWISE；DP>1 时动态 k 自动关闭 |

## 3. CUDA Graph 尺寸

| 位置 | 代码 | 说明 |
|---|---|---|
| config/vllm.py 2416–2650 | `_set_cudagraph_sizes` | 上限 `min(seqs×(1+k)×2, 512 或 SM100 的 1024)`；默认网格 ∪ `n×(1+k)` |
| config/compilation.py 1537 | `adjust_cudagraph_sizes_for_spec_decode` | 只在 MRV1 + FULL decode + qlen>1：尺寸取整到 (1+k) 的倍数，超上限丢弃（issue 28207） |
| v1/worker/gpu/cudagraph_utils.py 259–350 | `_init_candidates` | MRV2：每个 query 长度一组 uniform decode 图，动态 k 下每档一组 |

## 4. 草稿（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| v1/spec_decode/ngram_proposer.py 210 | `_find_longest_matched_ngram_and_propose_tokens` | 反转序列上跑 KMP 前缀函数，找最长后缀匹配（同长取最早） |
| v1/spec_decode/suffix_decoding.py | | 跨请求后缀树（arctic-inference） |
| v1/spec_decode/eagle.py 10 | `EagleProposer` | EAGLE / EAGLE-3 / MTP 共用；MRV2 对应 worker/gpu/spec_decode/eagle/ |
| v1/spec_decode/metrics.py 121 | `SpecDecodingLogging.log` | "SpecDecoding metrics:" 接受长度、per-position、接受率 |

## 5. 验证（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| v1/sample/rejection_sampler.py 432 | `rejection_sample` | V1：greedy 比 argmax；random 走 `rejection_random_sample_kernel`（896） |
| 同上 745 | `sample_recovered_tokens` | 每请求一组 Exp(1) 噪声，argmax(max(p-q,0)/E) |
| v1/worker/gpu/spec_decode/rejection_sampler.py 178 | `use_block_verification` | `block` 只有 MRV2 认 |
| v1/worker/gpu/spec_decode/rejection_sampler_utils.py 335 / 418 / 520 | 累积 log p / 残差质量 / `_rejection_kernel` | standard、block、synthetic、fly 同一个 kernel |
| v1/spec_decode/fly.py 16 | `compute_fly_entropy` | top-k 熵 |
| v1/worker/gpu/spec_decode/adaptive_verification.py 128 / 227 | `assign_draft_token_budget` / Manager | 存活概率排序、全局预算 |

## 6. sglang 对照

| 位置 | 代码 | 说明 |
|---|---|---|
| srt/arg_groups/spec.py 109–120 | `speculative_accept_threshold_single/_acc`、`use_rejection_sampling`、`use_block_verification` | 阈值默认 1.0；后两者只支持 topk=1 |
| 同上 237 / 241 | `speculative_adaptive` / `_config` | EMA 按 BS 档调 num_steps |
| srt/speculative/adaptive_spec_params.py 28 / 153 | `DEFAULT_ADAPTIVE_CONFIG` / `AdaptiveStepSlot` | 四档 1/8/32/64；warmup 10、每 5 批重算、ema_alpha 0.2 |
| srt/speculative/speculative_hook.py 448 / 1471 | 并发重置为 48 / 自动 (steps, topk, draft_tokens) | Llama (5,4,8)；DeepSeek/Qwen3 (3,1,4) |
| srt/speculative/eagle_utils.py 503 / 745 | greedy 路径 / 采样路径 | 默认 `tree_speculative_sampling_target_only`；开 rejection sampling 走 `chain_speculative_sampling_triton` |
| jit_kernel/.../speculative/sampling.cuh | `TreeSpeculativeSamplingTargetOnly` | `coin < prob_acc/th_acc 或 p ≥ th_single`；拒绝 token 在残差里置零 |
| srt/speculative/ngram_worker.py | NGRAM | 树：draft tokens 默认 12 |

## 7. 一句话对照

| 需求 | vllm | sglang |
|---|---|---|
| EAGLE-3 k=3 | `-sc '{"method":"eagle3","model":D,"num_speculative_tokens":3}'` | `--speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |
| 按并发调 k | 静态表 `num_speculative_tokens_per_batch_size` | `--speculative-adaptive`，运行时 EMA |
| 真正用草稿概率 | `draft_sample_method=probabilistic` | `--speculative-use-rejection-sampling` |
| 块验证 | `rejection_sample_method=block`（MRV2） | `--speculative-use-block-verification` |
| 有损加速 | `fly` / `synthetic` | 阈值 < 1 |
