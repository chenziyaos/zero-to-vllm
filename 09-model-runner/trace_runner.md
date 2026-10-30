# Model Runner、torch.compile 与 CUDA Graph：逐行对照

基线 vllm 主干 0d6f92a，sglang 主干 8fabf5e。

## 1. 模式与档位（vllm/config/compilation.py、vllm/config/vllm.py）

| 行 | 代码 | 说明 |
|---|---|---|
| compilation 37 | `CompilationMode` | NONE 0 / STOCK_TORCH_COMPILE 1 / DYNAMO_TRACE_ONCE 2 / VLLM_COMPILE 3；只有 3 会切图、缓存、做自定义 pass |
| compilation 53 | `CUDAGraphMode` | NONE 0、PIECEWISE 1、FULL 2；FULL_DECODE_ONLY = (FULL, NONE)，FULL_AND_PIECEWISE = (FULL, PIECEWISE)，元组是 (decode, mixed) |
| compilation 77–85 | `requires_piecewise_compilation` / `without_piecewise` | PIECEWISE → NONE，F&P → FDO；`separate_routine` 在元组模式下为真 |
| compilation 502 | `splitting_ops` | None 表示用注意力算子切；`[]` 表示整图不切 |
| compilation 578 / 583 | `compile_sizes` / `compile_ranges_endpoints` | 前者对具体尺寸再特化编译，后者把 token 数切成区间各编一份 |
| compilation 610–650 | docstring | 五种模式各自的取舍 |
| compilation 672 | `use_inductor_graph_partition` | 让 Inductor 在自己的图里切分，不走 `split_graph` |
| compilation 695 | `max_cudagraph_capture_size` | 用户给了就以它为上限 |
| compilation 768 | `_attention_ops` | 默认的切图算子表 |
| compilation 1162–1212 | `set_splitting_ops_for_v1` | 不开 inductor partition 时把 kv_cache_update 也加进切点（issue #33267，同时关掉 fuse_rope_kvcache）；引用 PR #20059 |
| compilation 1376 | `resolve_cudagraph_mode_and_sizes` | 按 AttentionCGSupport 降级：mixed 不能 FULL → F&P 或 FDO；decode 也不能 → PIECEWISE 或 NONE；q>1 且 < UNIFORM_BATCH → PIECEWISE；没有 piecewise 子图 → `without_piecewise()` |
| compilation 1537 | `adjust_cudagraph_sizes_for_spec_decode` | MRv1 + FULL decode + q>1：尺寸取 q 的倍数，空了退成 [q]（issue #28207） |
| compilation 1567 | `get_compile_ranges` | endpoints → Range(1, e0)、Range(e0+1, e1)… |
| vllm 147 / 280–360 / 450 | `OptimizationLevel` | O0 不编译不捕获；O1 Dynamo + Inductor + PIECEWISE；O2（默认）加融合 + FULL_AND_PIECEWISE；O3 目前同 O2 |
| vllm 746 / 3071 | `use_v2_model_runner` | 支持时默认走 MRv2；不支持的特性列表在 3071 |
| vllm 1144 | `_normalize_piecewise_cudagraph_mode` | 编译模式不是 VLLM_COMPILE 又没开 breakable 时 `without_piecewise()` |
| vllm 2470–2625 | `_set_cudagraph_sizes` | 默认上限 min(2 × seqs × q, 512)（Blackwell 1024），再被 max_num_batched_tokens 截；[1,2,4] + 8..248/8 + 256..max/16 |
| vllm 2676 | `_set_compile_ranges` | 默认 endpoints = [max_num_batched_tokens] |

## 2. 切图与编译（vllm/compilation/）

| 行 | 代码 | 说明 |
|---|---|---|
| backends.py 576 | `split_graph` | 遇到切点 subgraph_id 加一，连续切点放一起；getitem 跟着输入走；`split_module(keep_original_order=True)` |
| backends.py ~650 | `wrap_with_cudagraph_if_needed` | 每个非切点子图包一层 `CUDAGraphWrapper(PIECEWISE)`；piecewise 关闭或 inductor partition 时跳过 |
| backends.py 703–705 | first / last graph | 第一段和最后一段各自处理 gc 与输出弱引用 |
| backends.py 311 / 418 | 日志 | "Directly load the compiled graph(s) for compile range" / "Compiling a graph for compile range %s takes %.2f s" |
| backends.py 828 | `VllmBackend` | Dynamo 回调入口：切图 → 每段交给 PiecewiseBackend |
| backends.py 1076–1107 | 缓存目录 | factors = [env, config, code, compiler]，code 是被追踪文件的 sha256；`sha256(str(factors))[:10]`；`$VLLM_CACHE_ROOT/torch_compile_cache/{hash}/rank_{r}_{dp}/{prefix}` |
| backends.py 1126 | 日志 | "Using cache directory: %s for vLLM's torch.compile" |
| piecewise_backend.py 86 / 357 | `PiecewiseBackend` | 按 compile range 选编译产物，首次命中时编译 |
| monitor.py 20 | 日志 | "torch.compile took %.2f s in total" |
| cuda_graph.py 235 | `CUDAGraphWrapper.__call__` | 读 forward context 里的运行时模式和 BatchDescriptor；模式不符就直接调 runnable；首次见到这个描述符就捕获（gc 关闭、输出存弱引用），之后 replay |
| breakable_cudagraph.py 247 | `BreakableCUDAGraphWrapper` | 受 sglang PR #19102 启发；`VLLM_USE_BREAKABLE_CUDAGRAPH` 默认关（envs.py 182 / 783） |

## 3. 注意力后端能力（vllm/v1/attention/backend.py、docs/design/cuda_graphs.md）

| 位置 | 代码 | 说明 |
|---|---|---|
| backend.py 570 | `AttentionCGSupport` | ALWAYS 3 > UNIFORM_BATCH 2 > UNIFORM_SINGLE_TOKEN_DECODE 1 > NEVER 0 |
| cuda_graphs.md 175–187 | 后端表 | FA3、Triton：ALWAYS；FA2、AITER FA、FlashMLA、FlashInferMLA：UNIFORM_BATCH；FlashInfer：UNIFORM_SINGLE_TOKEN_DECODE（Blackwell + TRTLLM 时 UNIFORM_BATCH）；AITER MLA、CUTLASS MLA、Mamba：UNIFORM_SINGLE_TOKEN_DECODE |
| gpu_model_runner.py 6760 | `_check_and_update_cudagraph_mode` | 取所有 KV 组、所有后端的最小能力，调 resolve，再初始化 dispatcher 键（drafter 也一份） |

## 4. Dispatcher（vllm/v1/cudagraph_dispatcher.py）

| 行 | 代码 | 说明 |
|---|---|---|
| 71 | `_compute_bs_to_padded_graph_size` | 0..max 每个 bs 查表到 ≥ bs 的最小捕获尺寸；同时校验 compile_sizes |
| ~100 | `_get_lora_cases` | 无 LoRA [0]；specialize（默认）[0] + 捕获数；`specialize_active_lora` 时捕获数 = 2 的幂 + max_loras+1，否则只有 max_loras+1 |
| 130 | `_create_padded_batch_descriptor` | uniform decode 且有 FULL 时 num_reqs = padded // q（FA3 的 scheduler_metadata 依赖精确请求数） |
| 164 | `initialize_cudagraph_keys` | mixed 模式每个尺寸一个键；separate_routine 时另加 uniform decode 的 FULL 键（尺寸落在 [q, q × max_num_seqs]） |
| 233 | `dispatch` | 超过最大尺寸直接 NONE；LoRA 数按捕获表上取整；先查 FULL 键，再把 num_reqs / uniform 抹掉查 PIECEWISE 键，都没有就 NONE |
| 324 | `get_capture_descs` | PIECEWISE 在前、FULL 在后，各自从大到小 |

## 5. Model Runner V1（vllm/v1/worker/gpu_model_runner.py，7192 行）

| 行 | 代码 | 说明 |
|---|---|---|
| 1924 | `_prepare_inputs` | 把 scheduler 输出拼成 input_ids / positions / slot_mapping / query_start_loc |
| 3586 | `_model_forward` | 真正调模型 |
| 3684 | `_determine_batch_execution_and_padding` | 调 dispatcher，决定运行时模式和补齐后的 token 数；DP 下各 rank 对齐 |
| 3906 | `execute_model` | 一步的主体 |
| 4155 | `set_forward_context` | 把 attn_metadata、cudagraph 模式、BatchDescriptor 放进全局上下文，wrapper 从这里读 |
| 4270 | `sample_tokens` | 采样，与前向分开，便于异步调度重叠 |
| 4967 | `load_model` | 加载权重后 5125–5165 按模式包外层：BreakableCUDAGraphWrapper / CUDAGraphWrapper(FULL) / UBatchWrapper |
| 5447 | `_dummy_run` | profile 与捕获都走这里，用假输入跑指定 BatchDescriptor |
| 6072 | `profile_run` | 最大 batch 跑一次测激活峰值 |
| 6268 | `profile_cudagraph_memory` | 临时池里每种模式只捕获最大的两个描述符，估算 = 第 1 个 + (n−1) × max(第 2 个, 1 MiB)；日志 "Profiling CUDA graph memory: PIECEWISE=… FULL=…" |
| 6417 / 6571 | `capture_model` / `_capture_cudagraphs` | 按 `get_capture_descs` 顺序捕获 |
| 6517 | 日志 | "Graph capturing finished in %.0f secs, took %.2f GiB" |

## 6. Model Runner V2（vllm/v1/worker/gpu/）

| 位置 | 代码 | 说明 |
|---|---|---|
| cudagraph_utils.py 71 | `BatchExecutionDescriptor` | MRv2 的键 |
| cudagraph_utils.py 150 / 259 | `CudaGraphManager` / `_init_candidates` | 每种模式生成候选尺寸，投机解码的倍数在这里处理（PR #45953） |
| cudagraph_utils.py 427 / 529 | `capture` / `dispatch` | |
| cudagraph_utils.py 561 / 576 / 584 | `run_fullgraph` / `init_breakable_cg_runner` / `run_pw_graph` | FULL、breakable、piecewise 三条路径 |
| cudagraph_utils.py 592 / 928 / 1037 | `ModelCudaGraphManager` / `profile_cudagraph_memory` / `_extrapolate_full_graph_memory` | |
| model_runner.py 770–786 | `piecewise_capture_available` | 开了 breakable 或模型里有编译过的子模块才算有 piecewise 可用 |

## 7. 命令行（vllm/engine/arg_utils.py）

| 行 | 参数 | 说明 |
|---|---|---|
| 1001 | `--enforce-eager` | 不编译、不捕获 |
| 1829 / 1833 | `--cudagraph-capture-sizes` / `--max-cudagraph-capture-size` | |
| 1896 | `--compilation-config` / `-cc` | JSON，或 `-cc.cudagraph_mode=PIECEWISE` 点号写法 |
| 1916 | `--optimization-level` | `-O0`…`-O3` 由 argparse_utils.py:395 改写 |
| 1918 | `--performance-mode` | balanced / interactivity / throughput；interactivity 把 1–32 每个都捕获 |

## 8. SGLang 对照

| 位置 | 代码 | 说明 |
|---|---|---|
| cuda_graph_hook.py 480 | prefill 尺寸 | 4..32/4、48..256/16、288..512/32、576..1024/64、1280..4096/256、4608..max/512（token） |
| cuda_graph_hook.py 500 | decode 尺寸 | [1,2,4,8,12] + 16..256/8 + 272..511/16 + 512..max/32；投机解码另一套更密；`--disable-cuda-graph-padding` 时 1..max 全捕获 |
| memory_hook.py 89 | 显存档位 | <20G 2048/8；<35G 4096/48（tp≥4 160）；<60G 4096/32（160）；<90G 与 <160G 8192/256（512）；更大 16384/512；未知 4096/160 |
| cuda_graph_config.py 123 / 150 | 后端默认 | prefill 在 CUDA 上默认 breakable，decode 默认 full |
| breakable_cuda_graph_backend.py / breakable_cuda_graph.py | breakable graph | 把不能进图的段落在 eager 里跑，其余段落各自是一张图 |
| decode_cuda_graph_runner.py 252 / 395 | `DecodeCudaGraphRunner` / `get_batch_sizes_to_capture` | |
| decode_cuda_graph_runner.py 877 | `can_run_graph` | bs 超过最大捕获尺寸、或有不支持的特性时退回 eager |
| decode_cuda_graph_runner.py 1281 / 1325 / 1358 | `capture` / `freeze_gc` / reversed | 从大到小捕获，捕获期间冻结 gc |
| decode_cuda_graph_runner.py 1401 / 1659 | `capture_one_shape` / `_pad_to_bucket` | bisect 到桶 |
| f_exec_.py ~600–705 | 参数 | `--cuda-graph-backend-decode/prefill`（full / breakable / disabled）、`--cuda-graph-max-bs-*`、`--cuda-graph-bs-*`、`--disable-cuda-graph-padding`、`--enable-profile-cuda-graph`、`--enable-torch-compile`（实验，`--torch-compile-max-bs` 32） |
