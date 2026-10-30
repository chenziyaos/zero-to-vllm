# 并行：源码位置与参数对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。表里的“行”都是这两个提交的行号。

## 1. 分组（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| config/parallel.py 983 起 | `ParallelConfig.__post_init__` | `world_size = pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size`；DP 不算在 world_size 里，是另外的引擎 |
| distributed/parallel_state.py 2150–2370 | `initialize_model_parallel` | `all_ranks = arange(world).reshape(-1, DP, PP, PCP, TP)`；要哪个维度的组，就把它 transpose 到最后再 `reshape(-1, size)` |
| 同上 2253–2258 | DCP 组 | `dcp > 1` 时先 `transpose(-1, -2)`（先跨 PCP 再跨 TP），再 `reshape(-1, dcp)`；PCP=1 时 DCP 组就是 TP 组里相邻的 dcp 张卡 |
| 同上 2271 / 2291 / 2307 | PCP / PP / DP 组 | `transpose(3,4)` / `transpose(2,4)` / `transpose(1,4)` |
| 同上 2331 | EP 组 | `transpose(1, 2).reshape(-1, DP*PCP*TP)`；稠密模型不建 EP 组；开 EPLB 时再建一个同成员的 EPLB 组 |
| model_executor/layers/fused_moe/config.py 1120 | `FusedMoEParallelConfig.make` | 开 `-ep` 时 `ep = tp * dp * pcp`、MoE 层内 tp=1；不开时 MoE 按 `tp * dp` 做张量并行 |
| v1/engine/core.py 1594–1603 | 引擎类选择 | 只有 MoE + DP 用 `DPEngineCoreProc`；稠密模型的 DP rank 调 `reconfigure_for_independent_dp_rank()` 当独立引擎跑 |
| config/parallel.py 1050–1054 | 离线 DP 校验 | 稠密模型离线 DP 报 `Offline data parallel mode is not supported/useful for dense models.` |

## 2. 参数（vllm `vllm serve`）

| 参数 | 缩写 | 默认 | 说明 |
|---|---|---|---|
| `--tensor-parallel-size` | `-tp` | 1 | 每层切到 tp 张卡，每层两次 all-reduce |
| `--pipeline-parallel-size` | `-pp` | 1 | 按层切段；余数层给倒数第 2、3… 段（`get_pp_indices`），`VLLM_PP_LAYER_PARTITION` 可覆盖 |
| `--data-parallel-size` | `-dp` | 1 | 整份模型的副本数；MoE 下注意力 DP、专家 EP |
| `--decode-context-parallel-size` | `-dcp` | 1 | KV 按 token 交错切到 dcp 张卡；必须整除 tp |
| `--enable-expert-parallel` | `-ep` | off | MoE 层改为专家并行，EP 组 = DP×PCP×TP |
| `--all2all-backend` | | `allgather_reducescatter` | 另有 `deepep_high_throughput` / `deepep_low_latency` / `mori_*` / `nixl_ep` / `flashinfer_nvlink_*` 等 |
| `--enable-eplb` / `--eplb-config` | | off | `window_size=1000`、`step_interval=3000`、`num_redundant_experts=0`、`use_async=True`、`policy=default` |
| `--enable-dbo` | | off | 两个 microbatch 交错计算与 a2a；`dbo_decode_token_threshold=32`、`dbo_prefill_token_threshold=512` |
| `--data-parallel-size-local` 等 | `-dpl` `-dpr` `-dpa` `-dpp` `-dpb` | | 多机 DP：本机 rank 数、起始 rank、地址、RPC 端口、后端 |
| `--data-parallel-hybrid-lb` / `--data-parallel-external-lb` | `-dph` / `-dpe` | off | 负载均衡交给外部路由（每个 rank 一个端口） |
| `--expert-placement-strategy` | | `linear` | `round_robin` 只在无冗余专家时有意义 |

## 3. 校验（vllm）

| 位置 | 条件 | 报错 / 行为 |
|---|---|---|
| config/parallel.py 542–630 | `enable_eplb` 但没开 EP | 报错 |
| 同上 | `num_redundant_experts > 0` 但没开 EPLB | 报错 |
| 同上 | PCP=1 时 `tp % dcp != 0` | 报错 |
| config/model.py 1585–1613 | GQA 且 `tp <= kv_heads` | `requires --tensor-parallel-size (tp) > KV heads (kv)` |
| 同上 | `dcp > tp // kv_heads` | `exceeds the maximum supported value` |
| 同上 | `(q_heads / kv_heads) % dcp != 0` | 报错 |
| config/model.py 1690 | 每卡 KV 头数 | `max(1, kv_heads // tp)`；MLA 恒为 1（每卡存完整 latent） |
| distributed/eplb/eplb_state.py 850 | `num_gpus % num_nodes != 0` | 告警并退回全局策略（num_nodes=1） |

## 4. DP 的同步（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| v1/engine/core.py 2567–2650 | `DPEngineCoreProc.run_busy_loop` | 本步没执行、但全局还有活 → `execute_dummy_batch`；全局没活 → `wave_complete`、`current_wave += 1` |
| 同上 2656 | `_has_global_unfinished_reqs` | 第 1 步和每 `dp_sync_interval`（16）步才同步，其余步直接返回 True |
| config/parallel.py 889–914 | `sync_dp_state` | CPU 组上一次 SUM all-reduce：`[has_unfinished, pending_pause]` |
| v1/worker/gpu_worker.py 1469 | `execute_dummy_batch` | `_dummy_run(1, uniform_decode=True)` |
| v1/worker/dp_utils.py 122–187 | `_synchronize_dp_ranks` | 一次 all-reduce 拿到各 rank 的 token 数、是否 ubatch、CG 模式；CG 取最小；不是 NONE 就把所有 rank 补到最大值 |
| v1/worker/ubatch_utils.py 129–148 | DBO 门槛 | decode ≥ 32、prefill ≥ 512 token 才切 microbatch；最后一个 microbatch 为空则放弃 |

## 5. EPLB（vllm default policy）

| 步骤 | 函数 | 做什么 |
|---|---|---|
| 统计 | eplb_state.py 697–716 | 每次重排前只记录最后 `window_size` 步的每专家 token 数 |
| 时机 | eplb_state.py 466–470、666–693 | 计数器初值 `step_interval - step_interval // 4`（2250），每步 +1，到 `step_interval` 触发重排并清零：首次在启动后第 750 步，之后每 3000 步 |
| 1 | `balanced_packing(组负载, num_nodes)` | 专家组（DSV3 n_group=8）贪心装进节点，组内专家同节点，减少跨节点流量 |
| 2 | `replicate_experts(节点内负载, 节点内物理槽)` | 每次给 `负载 / 副本数` 最大的逻辑专家加一个副本 |
| 3 | `balanced_packing(物理专家负载, 每节点 GPU 数)` | 副本装进节点内 GPU，每卡槽数相同 |
| 4 | `preserve_intragpu_slots` | 同一张卡上已存在的专家不挪槽，少拷权重 |
| 路由 | base_router.py 45–60 | token 选副本：`(token_idx * 2654435769) & 0xFFFFFFFF % 副本数` |
| 指标 | eplb_state.py 66–71 | `balancedness = Σ_层 平均卡负载 / Σ_层 最忙卡负载` |

## 6. SGLang 对照

| 意图 | vLLM | SGLang | 差别 |
|---|---|---|---|
| 张量并行 | `-tp 8` | `--tp 8` | 一样 |
| 卡数口径 | world = PP×TP×PCP，DP 另乘 | `tp_size` 就是总卡数，attn DP 在 TP 组内切 | `vllm -tp 8 -dp 8` 是 64 卡；`sglang --tp 8 --attn-dp-size 8` 是 8 卡 |
| 注意力 DP + 专家 EP | `-dp 8 -ep` | `--tp 8 --attn-dp-size 8 --ep-size 8` | SGLang 的 FFN 可以留在 TP（`moe_dense_tp_size`） |
| a2a 后端 | `--all2all-backend deepep_low_latency` | `--moe-a2a-backend deepep` | SGLang 这类后端会把 `ep_size` 强制改成 `tp_size`（overrides.py 1612–1621） |
| EPLB | `--enable-eplb --eplb-config '{"num_redundant_experts":32}'` | `--enable-eplb --ep-num-redundant-experts 32` | 节奏：vLLM 每 3000 步（首次在第 750 步）；SGLang `eplb_rebalance_num_iterations=1000` |
| overlap | `--enable-dbo` | `--enable-two-batch-overlap` | 门槛：vLLM decode 32 token；SGLang `tbo_token_distribution_threshold=0.48` |
| DCP | `-dcp 8` | `--dcp-size 8` | SGLang 默认 `dcp_comm_backend` 在 CUDA 上是 a2a；vLLM 默认 `ag_rs` |
| 注意力 DP 下的 chunk | 每个 DP rank 各自 `max_num_batched_tokens` | `chunked_prefill_size //= attn_dp_size`，`schedule_conservativeness × 0.3` | SGLang 的 chunk 是全组总量 |
| lm_head | DP rank 各持一份（tp=1） | `--enable-dp-lm-head` 在 attn TP 组内切词表 | 没有 DP 时 SGLang 自动关掉 |
