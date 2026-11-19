# 结构化输出与多 LoRA：源码位置与参数对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。表里的"行"都是这两个提交的行号。vllm 路径相对 `vllm/`，结构化输出在 `v1/structured_output/`（下文简称 `so/`）。sglang 路径相对 `python/sglang/srt/`。

## 1. 结构化输出：入口与后端

| 位置 | 代码 | 说明 |
|---|---|---|
| config/structured_outputs.py 23 | `backend = "auto"` | 可选 auto / xgrammar / guidance / outlines / lm-format-enforcer |
| config/structured_outputs.py 30、67 | `disable_any_whitespace` | 只有 xgrammar、guidance 支持 |
| config/structured_outputs.py 39、45 | `reasoning_parser`、`enable_in_reasoning` | 有推理解析器时默认推理段不约束；`enable_in_reasoning=True` 从第一个 token 起约束 |
| sampling_params.py 39、44 | 嵌套上限 64，structural tag 再加 16 | 超过直接拒绝 |
| sampling_params.py 217–260 | `_exceeds_json_nesting` | 按层广度优先，每层只保留 dict/list 子节点 |
| sampling_params.py 1377–1444 | auto 选择 | Tekken + Lark → guidance；xgrammar 能吃就 xgrammar；否则 Mistral 非 Tekken 或含 patternProperties → outlines，慢 tokenizer 报错，其余 guidance |
| so/backend_xgrammar.py 362–476 | `has_xgrammar_unsupported_json_features` | multipleOf、uniqueItems/contains、未知 format、pattern+长度、propertyNames 组合、patternProperties+properties、多个 patternProperties、allOf ≥ 2 |
| so/backend_guidance.py 93–116 | `has_guidance_unsupported_json_features` | 任意一层出现 patternProperties |
| so/\_\_init\_\_.py 143–150 | 一个引擎一个后端 | auto 下第一个结构化请求决定整台实例的后端，之后不同的报错 |

## 2. 编译、bitmask 与重叠

| 位置 | 代码 | 说明 |
|---|---|---|
| so/\_\_init\_\_.py 81–96 | 线程池 | 填 mask 阈值 128、每批 16、`max(1, min(cpu//2, 8))` 线程；编译 `max(1, (cpu+1)//2)` 线程 |
| so/\_\_init\_\_.py 46–47 | `_CONSTRAINED_FROM_START=(0,0)`、`_NO_SPEC_TOKENS=(1,1)` | `_get_constraint_bounds` 的两个常用返回 |
| so/\_\_init\_\_.py 231–540 | `_fill_bitmasks` / `trim_spec_tokens` / `grammar_bitmask` / `accept_tokens` | 投机时逐 draft accept、每位置一行、最后 rollback；推理段的行屏蔽语法不能开头的隐式结束符 |
| so/backend_xgrammar.py 202 | `max_rollback_tokens = num_speculative_tokens` | |
| so/utils.py 48、62 | `strip_speculative_padding`；`compile_regex_with_timeout` | 正则编译超时 `VLLM_REGEX_COMPILATION_TIMEOUT_S=5`（envs.py 232） |
| envs.py 1717 | `VLLM_XGRAMMAR_CACHE_MB=512` | |
| v1/engine/core.py 685–693 | `execute_model(non_block=True)` → bitmask → `sample_tokens` | CPU 填 mask 与 GPU 前向重叠 |
| v1/worker/gpu/structured_outputs.py 12、15、70–171 | 块 8192；`grammar_invalid_drafts`；`_apply_grammar_bitmask_kernel` | bit 为 0 的 logit 置 -inf；超出可接受 draft 数的位置整行作废 |

## 3. 多 LoRA

| 位置 | 代码 | 说明 |
|---|---|---|
| config/lora.py 32–67 | `max_lora_rank=16`、`max_loras=1`、`max_cpu_loras=None`（取 max_loras）、`fully_sharded_loras=False` | |
| lora/layers/base_linear.py 104–155 | A/B 预分配 | 按 `max_loras × max_lora_rank` 留，与实际 adapter rank 无关 |
| lora/model_manager.py 1406、1449–1462 | LRU activate | 槽满淘汰最久未用的；没有可淘汰的报 "No free adapter slots." |
| v1/core/sched/scheduler.py 1038–1047 | `scheduled_loras` | running 里所有 `lora_int_id > 0` 的请求，含本步没排上的 |
| v1/core/sched/scheduler.py 1091–1102、1565 | waiting 准入 | 槽满且是新 adapter 就 skip |
| v1/core/kv_cache_utils.py 589–607 | `_gen_lora_extra_hash_keys` | `("lora", name, path)` |
| envs.py 128 | `VLLM_ALLOW_RUNTIME_LORA_UPDATING=False` | 打开才有 `/v1/load_lora_adapter`、`/v1/unload_lora_adapter` |
| v1/metrics/loggers.py 1096–1110 | `vllm:lora_requests_info` | running / waiting adapter 名单、max_lora |

## 4. 与 SGLang 的参数对照

| 维度 | vLLM | SGLang |
|---|---|---|
| 选后端 | `--structured-outputs-config.backend auto`，逐请求判定，引擎锁首个 | `--grammar-backend`（默认 xgrammar），启动定死 |
| 推理段 | `reasoning_parser` + `enable_in_reasoning` | `--reasoning-parser`，`think_end_id` 前不约束 |
| 填 mask 时机 | engine core 在前向发出后算 | overlap 下 `delay_sample_func`（tp_worker.py 728–749）：前向先发，采样延后；`update_regex_vocab_mask`（sampling_batch_info.py 382–407）用 `fill_vocab_mask_batched` 填整批 |
| 编译超时 | 仅正则 5 s | grammar_manager.py 255–300：轮询 ≥ 10000 次判失败，间隔 max(0.005 s, 步长) |
| 投机回滚 | `max_rollback_tokens = k` | `MAX_ROLLBACK_TOKENS=200`（xgrammar_backend.py 63） |
| jump-forward | 无 | `try_jump_forward` 在，调度路径没有调用 |
| LoRA 槽数 | `--max-loras 1`，base 不占槽 | `--max-loras-per-batch 8`，base 占一个 |
| CPU 侧缓存 | `--max-cpu-loras` | `--max-loaded-loras`，`--lora-eviction-policy lru/fifo` |
| 准入 | running 里的 adapter 占槽，满了 skip | `validate_lora_batch`（lora_manager.py 427–456）+ pinned 预留 |
| 防饿死 | 无 | `--lora-drain-wait-threshold`（默认 0 关闭），容差 1.2 |
| kernel | Triton shrink/expand（Punica 风格） | `--lora-backend csgmv`（默认）/ triton / torch_native / triton_v2 |
| 前缀隔离 | `("lora", name, path)` | extra_key 拼 lora_id（schedule_batch.py 1170–1173）；`--lora-paths` 用 uuid5(name\0path)，动态加载 uuid4 |
| 异步加载 | — | `--enable-lora-overlap-loading` |
