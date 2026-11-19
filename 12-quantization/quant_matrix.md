# 量化：源码位置与参数对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。表里的“行”都是这两个提交的行号。

## 1. 注册与配置（vllm）

| 位置 | 代码 | 说明 |
|---|---|---|
| model_executor/layers/quantization/__init__.py 15 | `QuantizationMethods` | 所有 `--quantization` 合法值的 Literal |
| 同上 52 / 61 / 112 | 废弃名单、`register_quantization_config`、`get_quantization_config` | 插件可以注册自己的量化方法；`awq` / `awq_marlin` 等旧名映射到统一实现 |
| config/quantization.py 195–241 | 在线量化简写 | `fp8_per_tensor` / `fp8_per_block` / `fp8_per_channel` / `mxfp8` / `mxfp4` 作用于线性层 + MoE；`int8_per_channel_weight_only`、`nvfp4_per_token` 只作用于 MoE；`mxfp8` / `mxfp4` 在 checkpoint 自带量化元数据时不生效 |
| layers/quantization/fp8.py 92 / 106–111 | `Fp8Config` | `--quantization fp8` 只读 FP8 checkpoint；对 BF16 模型报 "no longer supports online quantization. Please use `--quantization fp8_per_tensor`" |
| 同上 240 / 349 / 356 | `Fp8LinearMethod` | 决定激活量化 key，调 `init_fp8_linear_kernel`；`use_marlin` 标记 W8A16 路径 |
| config/vllm.py 945–949 | 最低算力校验 | FP8 75、ModelOpt FP8 80、ModelOpt NVFP4 75、ModelOpt MXFP8 80、MXFP4 80、AWQ 75、GPTQ 60、compressed-tensors 70 |

## 2. 线性层 kernel 选择（vllm `model_executor/kernels/linear/__init__.py`）

| 行 | 代码 | 说明 |
|---|---|---|
| 256 | `_get_linear_backend` | `kernel_config.linear_backend_per_quant[quant_key]` 优先于 `--linear-backend` |
| ~286 | `_LINEAR_BACKEND_KERNEL_MAP` | backend 名 → kernel 类集合 |
| 393 | `_filter_kernels_by_backend` | 没有匹配：WARNING 后返回完整列表（不报错） |
| 410 | `_resolve_backend_kernels` | 先 `prioritize_humming`，再按 backend 过滤 |
| 446 / 458 / 494 / 522 / 539 | INT8 / FP8 / FP8 block / W8A16 / WnA16 候选列表 | 顺序即优先级 |
| 568 / 592 / 615 / 625 | MXFP8 / NVFP4 / MXFP6 / MXFP4 候选列表 | |
| 648 | `is_supported_and_can_implement_kernel` | `is_supported()`（硬件、库）→ `can_implement()`（形状、group、dtype） |
| 746 / 887 / 1046 / 1191 | `init_fp8_linear_kernel` / `choose_mp_linear_kernel` / `init_mxfp4_linear_kernel` / `init_nvfp4_linear_kernel` | 各格式入口；NVFP4 W4A16 在 auto 下强制选（sm100/103 CuTe-DSL，非 sm90 一律 Marlin） |
| quantization/utils/humming/priority.py 11 / 27–49 | `_HUMMING_PREFERRED_CAPABILITIES = {90}` | 只在 SM90 把 Humming 挪到 Marlin 前 |

## 3. 形状约束

| 位置 | 约束 | 报错 |
|---|---|---|
| mixed_precision/marlin.py 58–67 | 行并行层每卡 K 必须整除 group | `in_features per partition {K} is not divisible by group_size = {g}.` |
| utils/fp8_utils.py 1296 | FP8 block：行并行每卡 K % block_k == 0 | `Weight input_size_per_partition = {} is not divisible by weight quantization block_k = {}.` |
| utils/fp8_utils.py 1347（fp8.py 555 调用） | MoE：每卡 intermediate % block_n，TP>1 时还要 % block_k | `validate_fp8_block_shape_moe` |
| utils/machete_utils.py 37 | Machete：K % 64、N % 128 | 不满足就跳到下一个 kernel |
| auto_awq.py 821 | 旧 AWQ 路径（AutoAWQLinearMethod） | `input size is not aligned` |

## 4. KV cache 量化

| 位置 | 代码 | 说明 |
|---|---|---|
| config/cache.py 44 | `CacheDType` | 22 个取值 |
| v1/kv_cache_interface.py 40–138 | `KVQuantMode` | NONE / FP8 per-tensor / per-token-head INT8/FP8/INT4 / NVFP4 等，决定每头每 token 的字节和 scale 布局 |
| layers/quantization/turboquant/config.py 20 / 129 / 151 / 160 | `TQ_PRESETS`、`key_packed_size`、`value_packed_size`、`slot_size` | head_dim 128：196 / 134 / 118 / 102 B |
| 同上 178–215 | `get_boundary_skip_layers` | 稠密模型首尾各 2 层保持 BF16；混合模型返回 [] |
| v1/attention/backends/mla/flashmla_sparse.py 79 / 112 | `fp8_ds_mla` / `nvfp4_ds_mla` | 每 token 656 B / 352 B |

## 5. vLLM 与 SGLang 参数对照

| 场景 | vLLM | SGLang |
|---|---|---|
| BF16 在线转 FP8 | `--quantization fp8_per_tensor` | `--quantization fp8` |
| 读 FP8 / AWQ / GPTQ / NVFP4 checkpoint | 自动识别 `quantization_config` | 自动；`--quantization` 与 checkpoint 不兼容时 `_verify_quantization`（model_config.py 2018）报错 |
| 指定 GEMM 后端 | `--linear-backend`、`kernel_config.linear_backend_per_quant` | `--fp8-gemm-backend`、`--fp4-gemm-backend`（exec_.py 276 / 285） |
| FP8 KV | `--kv-cache-dtype fp8`，scale 读 checkpoint，没有就 1.0（layers/quantization/kv_cache.py 110–162） | `--kv-cache-dtype fp8_e4m3` + `--quantization-param-path` |
| 4 bit KV | `nvfp4` / `int4_per_token_head` / `turboquant_*` | `nvfp4` / `fp4_mx_block16` |
| FP8 block auto 顺序 | FlashInfer+DeepGEMM（sm90）→ DeepGEMM → CUTLASS → b12x → Marlin → Humming → Triton → torch | DeepGEMM → Blackwell FlashInfer → sm120 CUTLASS → aiter → npu → xpu → Triton（fp8_utils.py 1108–1207） |
