# VLLM-09 Model Runner、torch.compile 与 CUDA Graph

运行环境：模拟器只要 CPU 和 Python 3 标准库；`gpu_cudagraph_modes.sh` 需要一张能放下 Qwen3-8B 的 GPU 和装好的 vLLM。基线 vllm 主干 0d6f92a，sglang 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_capture_sizes.py` | 复刻 `_set_cudagraph_sizes` 的默认捕获尺寸、`resolve_cudagraph_mode_and_sizes` 的降级规则、`CudagraphDispatcher` 的键与 dispatch、LoRA 键翻倍、编译缓存目录哈希和 compile range；SGLang 一侧有 decode / prefill 捕获尺寸生成器和按显存档位的默认值 |
| `test_sim_capture_sizes.py` | 22 个单测：尺寸个数与上限、interactivity、投机解码取倍数、用户尺寸过滤、SGLang 列表、降级表、dispatch 结果、LoRA 情况、padding、compile range |
| `gpu_cudagraph_modes.sh` | 真机实验：eager / -O0 / -O1 / 五种 cudagraph_mode 的 decode 延迟、启动耗时与显存、冷热编译缓存 |
| `trace_runner.md` | Model Runner、编译、CUDA graph 相关代码的逐行对照表（行号对应主干） |

## 用法

```bash
python3 sim_capture_sizes.py --scenario sizes      # 两边默认的捕获尺寸列表
python3 sim_capture_sizes.py --scenario resolve    # 后端能力 × 请求的模式 → 实际模式
python3 sim_capture_sizes.py --scenario dispatch   # 几个典型 batch 走哪张图、补多少
python3 sim_capture_sizes.py --scenario padding    # 不同 batch 分布下的 padding 浪费与走图比例
python3 sim_capture_sizes.py --scenario keys       # 每种模式要捕获多少张图、LoRA 翻倍、显存估算
python3 sim_capture_sizes.py --scenario cache      # 编译缓存目录怎么算、compile range 怎么切
python3 -m unittest test_sim_capture_sizes

bash gpu_cudagraph_modes.sh startup                # 每种模式起一次引擎，抓编译 / 捕获耗时
bash gpu_cudagraph_modes.sh latency                # 每种模式 × batch 1/8/64/256 的端到端延迟
bash gpu_cudagraph_modes.sh cache                  # 冷 / 热 / 关闭编译缓存
```

padding 场景里的 batch 分布是合成的：`decode 1–32 均匀` 代表低并发在线服务，`几何 均值 12` 是 running 请求数集中在个位数、偶尔冲高的形状，`1–256` 和 `1–1024` 代表离线批量。浪费 = 补出来的 token / 真实 token，只统计走图的 batch；走图比例 = batch 数落在最大捕获尺寸以内的占比。编译缓存场景里四个因子用示例字符串代替，哈希值本身没有意义，只看哪些改动会换目录。

## 期望输出

`--scenario sizes`：

```
== 1. vLLM 默认捕获尺寸（单位 token；decode 一个请求 1 个 token，所以也等于 decode 的请求数）
  H100 API server（max_num_seqs 1024，预算 8192）
    51 个，max 512：1, 2, 4, 8..256/8, 272..512/16
  A100 API server（256，预算 2048）
    51 个，max 512：1, 2, 4, 8..256/8, 272..512/16
  B200（1024，预算 16384，上限 1024）
    83 个，max 1024：1, 2, 4, 8..256/8, 272..1024/16
  H100 --performance-mode interactivity
    76 个，max 512：1..32/1, 40..256/8, 272..512/16
  H100 max_num_seqs 64
    19 个，max 128：1, 2, 4, 8..128/8
  H100 EAGLE 3 个草稿（query_len 4，256 条）
    49 个，max 512：4, 8..256/8, 272..512/16

== 2. SGLang 默认 decode 捕获尺寸（单位：请求数）与 prefill 捕获尺寸（单位：token）
  H100 80G tp1：chunked_prefill_size 8192，decode max_bs 256
    decode 36 个：1, 2, 4..16/4, 24..256/8
  H100 80G tp8：chunked_prefill_size 8192，decode max_bs 512
    decode 52 个：1, 2, 4..16/4, 24..256/8, 272..512/16
  B200 180G：chunked_prefill_size 16384，decode max_bs 512
    decode 52 个：1, 2, 4..16/4, 24..256/8, 272..512/16
  RTX 4090 24G：chunked_prefill_size 4096，decode max_bs 48
    decode 10 个：1, 2, 4..16/4, 24..48/8
  prefill（max_bs = chunked_prefill_size 8192）58 个：4..32/4, 48..256/16, 288..512/32, 576..1024/64, 1280..4096/256, 4608..8192/512
  投机解码 decode（max_bs 256）51 个：1..8/1, 10..32/2, 40..64/4, 72..256/8
```

`--scenario resolve`：

```
== 后端能力 × 请求的 cudagraph_mode → 实际模式（VLLM_COMPILE，splitting_ops 含注意力）
  后端                    AttentionCGSupport            FULL_AND_PIECEWISE    FULL                  FULL_DECODE_ONLY      投机解码 q=4
  FlashAttention 3      ALWAYS                        FULL_AND_PIECEWISE    FULL                  FULL_DECODE_ONLY      FULL_AND_PIECEWISE
  Triton                ALWAYS                        FULL_AND_PIECEWISE    FULL                  FULL_DECODE_ONLY      FULL_AND_PIECEWISE
  FlashAttention 2      UNIFORM_BATCH                 FULL_AND_PIECEWISE    FULL_AND_PIECEWISE    FULL_DECODE_ONLY      FULL_AND_PIECEWISE
  FlashMLA              UNIFORM_BATCH                 FULL_AND_PIECEWISE    FULL_AND_PIECEWISE    FULL_DECODE_ONLY      FULL_AND_PIECEWISE
  FlashInfer            UNIFORM_SINGLE_TOKEN_DECODE   FULL_AND_PIECEWISE    FULL_AND_PIECEWISE    FULL_DECODE_ONLY      PIECEWISE
  CUTLASS MLA / Mamba   UNIFORM_SINGLE_TOKEN_DECODE   FULL_AND_PIECEWISE    FULL_AND_PIECEWISE    FULL_DECODE_ONLY      PIECEWISE
  未列出的后端                NEVER                         PIECEWISE             ERROR                 PIECEWISE             PIECEWISE

  compilation mode 不是 VLLM_COMPILE（比如 mode=0 但仍要 CUDA graph）时没有 piecewise 子图可退：
    FlashAttention 3       FULL_AND_PIECEWISE → FULL_DECODE_ONLY
    Triton                 FULL_AND_PIECEWISE → FULL_DECODE_ONLY
    FlashAttention 2       FULL_AND_PIECEWISE → FULL_DECODE_ONLY
    FlashMLA               FULL_AND_PIECEWISE → FULL_DECODE_ONLY
    FlashInfer             FULL_AND_PIECEWISE → FULL_DECODE_ONLY
    CUTLASS MLA / Mamba    FULL_AND_PIECEWISE → FULL_DECODE_ONLY
    未列出的后端                 FULL_AND_PIECEWISE → NONE
```

`--scenario dispatch`：

```
== H100 默认：FULL_AND_PIECEWISE，max_num_seqs 1024，捕获尺寸 51 个，max 512
  decode 1 条                         → FULL      补到 1（+0），num_reqs=1，uniform=True
  decode 37 条                        → FULL      补到 40（+3），num_reqs=40，uniform=True
  decode 300 条                       → FULL      补到 304（+4），num_reqs=304，uniform=True
  decode 512 条                       → FULL      补到 512（+0），num_reqs=512，uniform=True
  decode 600 条                       → NONE   eager（超过 max 512），不补齐
  混合 37 token（1 个短 prefill + decode） → PIECEWISE 补到 40（+3），num_reqs=None，uniform=False
  混合 300 token                       → PIECEWISE 补到 304（+4），num_reqs=None，uniform=False
  混合 2048 token（chunked prefill）     → NONE   eager（超过 max 512），不补齐

== 同样的 batch，其他模式
  PIECEWISE            decode 37 → PIECEWISE 40；混合 37 → PIECEWISE 40；混合 300 → PIECEWISE 304
  FULL_DECODE_ONLY     decode 37 → FULL 40；混合 37 → NONE；混合 300 → NONE
  FULL                 decode 37 → FULL 40；混合 37 → FULL 40；混合 300 → FULL 304

== SGLang H100 tp1：decode max_bs 256，36 个尺寸
  decode   1 条 → 图 bs=1（+0）
  decode  37 条 → 图 bs=40（+3）
  decode 300 条 → 超过 256，不用图
```

`--scenario padding`：

```
== decode batch 的 padding 浪费（补的 token / 真实 token）与走图比例
  分布                  vLLM 默认 (51)            vLLM interactivity      vLLM max_num_seqs 64    SGLang tp1 (36)         SGLang tp≥4 (52)        
  decode 1–32 均匀       17.2% / 走图 100.0%      0.0% / 走图 100.0%     17.2% / 走图 100.0%     14.2% / 走图 100.0%     14.2% / 走图 100.0%    
  decode 几何 均值 12      19.8% / 走图 100.0%      2.0% / 走图 100.0%     19.8% / 走图 100.0%     14.9% / 走图 100.0%     14.9% / 走图 100.0%    
  decode 1–256 均匀       2.7% / 走图 100.0%      2.4% / 走图 100.0%      5.2% / 走图  50.0%      2.6% / 走图 100.0%      2.6% / 走图 100.0%    
  decode 1–1024 均匀      2.1% / 走图  50.0%      2.1% / 走图  50.0%      5.2% / 走图  12.5%      2.6% / 走图  25.0%      2.1% / 走图  50.0%    

== interactivity 只改 1–32 段：逐个 batch 看补齐
  bs=3   默认 → 4    interactivity → 3    SGLang → 4
  bs=5   默认 → 8    interactivity → 5    SGLang → 8
  bs=9   默认 → 16   interactivity → 9    SGLang → 12
  bs=13  默认 → 16   interactivity → 13   SGLang → 16
  bs=17  默认 → 24   interactivity → 17   SGLang → 24
  bs=33  默认 → 40   interactivity → 40   SGLang → 40
```

`--scenario keys`：

```
== 每种模式的 dispatcher 键（H100 默认 51 个尺寸，模型 36 层注意力）
   PIECEWISE 键每个对应一次分段前向：注意力把图切成约 层数 + 1 段，每段一个 torch.cuda.CUDAGraph
  PIECEWISE            PIECEWISE 键  51（≈ 1887 段图）  FULL 键   0
  FULL                 PIECEWISE 键   0（≈    0 段图）  FULL 键  51
  FULL_DECODE_ONLY     PIECEWISE 键   0（≈    0 段图）  FULL 键  51
  FULL_AND_PIECEWISE   PIECEWISE 键  51（≈ 1887 段图）  FULL 键  51

== LoRA：键数按 LoRA 情况翻倍（cudagraph_specialize_lora 默认 True）
  无 LoRA                               lora 情况 [0]  → PIECEWISE 51 + FULL 51
  max_loras 4                          lora 情况 [0, 5]  → PIECEWISE 102 + FULL 102
  max_loras 4，specialize_active_lora   lora 情况 [0, 1, 2, 4, 5]  → PIECEWISE 255 + FULL 255
  max_loras 16，specialize_active_lora  lora 情况 [0, 1, 2, 4, 8, 16, 17]  → PIECEWISE 357 + FULL 357

== 显存估算（profile_cudagraph_memory）：每种模式只真捕获最大的两个，其余线性外推
  PIECEWISE n=51：估算 = 第 1 个 + 50 × max(第 2 个, 1 MiB) ≥ 第 1 个 + 50 MiB
  FULL      n=51：估算 = 第 1 个 + 50 × max(第 2 个, 1 MiB) ≥ 第 1 个 + 50 MiB
  捕获顺序：PIECEWISE 先、FULL 后，各自从大到小，小图复用大图留下的显存池
```

`--scenario cache`：

```
== 编译缓存目录 = sha256(str([env, config, code, compiler]))[:10]（四个因子用示例字符串）
  基线                 ~/.cache/vllm/torch_compile_cache/726bfdad8e/rank_0_0/backbone
  换 max_num_seqs      ~/.cache/vllm/torch_compile_cache/52c514cb1b/rank_0_0/backbone
  改了模型文件一行     ~/.cache/vllm/torch_compile_cache/0663985943/rank_0_0/backbone
  升级 torch           ~/.cache/vllm/torch_compile_cache/20a60b1e9c/rank_0_0/backbone
  TP=2 的 rank 1       ~/.cache/vllm/torch_compile_cache/726bfdad8e/rank_1_0/backbone

== compile range：endpoints 切成左开右闭的区间，每个区间编译一份 Inductor 产物
  endpoints [8192]         → [1, 8192]
  endpoints [2048]         → [1, 2048]
  endpoints [512, 8192]    → [1, 512], [513, 8192]
```

## 真机上要核对的东西

- `startup`：`-O0` 和 `--enforce-eager` 不出现 "torch.compile took"；`-O1` 只有 PIECEWISE 那一行显存；默认 O2 两行都有。"Graph capturing finished" 的秒数在 5–20 s 区间是正常的。
- `latency`：batch 1 时 eager 和 FULL 的差距最大（CPU launch 开销占比最高）；batch 256 时差距缩小。FULL_DECODE_ONLY 和 FULL_AND_PIECEWISE 的 decode 延迟应该接近，差别在 prefill 混合步。
- `cache`：第二次启动日志出现 "Directly load the compiled graph(s)"，torch.compile 耗时降到个位数秒；`VLLM_DISABLE_COMPILE_CACHE=1` 时每次都重新 "Compiling a graph"。
