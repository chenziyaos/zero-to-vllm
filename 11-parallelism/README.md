# VLLM-11 并行：TP、PP、DP、EP 与 EPLB、DCP

运行环境：模拟器只要 CPU 和 Python 3 标准库；`serve_dp_ep.sh` 需要一台 8 卡 H100/H200（跑 DeepSeek-V3 要 8×H200 或两台 8×H100）和装好的 vLLM，SGLang 对照另装 sglang。基线 vllm 主干 0d6f92a，sglang 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_parallel.py` | 复刻 `initialize_model_parallel` 的 rank 分组、`get_pp_indices` 切层、每卡权重 / KV 账（GQA 复制 KV 头、MLA 每卡完整 latent、DCP 切 token）、TP all-reduce / EP a2a / PP send 的字节账、DCP 合法性校验、EPLB default policy（balanced_packing → replicate_experts → 节点内装箱）、DP wave 的 dummy batch 与 CUDA Graph 补齐、SGLang 参数对照 |
| `test_sim_parallel.py` | 19 个单测：分组与源码 docstring 一致、EP/DCP 组、切层、参数量、KV 头复制、MLA 的 KV 不随 TP 缩小、EPLB 单调改善、DP 补齐与 wave |
| `serve_dp_ep.sh` | 真机实验：TP8 / DP8+EP / EPLB / DBO / DCP / PP 六种起法 + bench，和 SGLang 的四种对照起法 |
| `parallel_matrix.md` | 分组、参数、校验、DP 同步、EPLB 步骤的源码行号，以及与 SGLang 的参数和语义对照 |

## 用法

```bash
python3 sim_parallel.py --scenario layout   # rank 排布：每种并行组里是哪些 GPU
python3 sim_parallel.py --scenario memory   # 每卡权重 / KV 余量 / 整组能放多少 token
python3 sim_parallel.py --scenario comm     # 每步通信次数和字节：TP all-reduce、EP dispatch/combine、PP send
python3 sim_parallel.py --scenario dcp      # DCP 合法性和每卡 KV 字节
python3 sim_parallel.py --scenario eplb     # EPLB 策略：倾斜负载下 balancedness 前后对比
python3 sim_parallel.py --scenario dp       # DP wave：dummy batch、每 16 步同步、CUDA Graph 补齐
python3 sim_parallel.py --scenario pp       # get_pp_indices 切层和每段显存
python3 sim_parallel.py --scenario sglang   # 同样的 8 卡，vLLM 和 SGLang 的参数怎么写、默认值差在哪
python3 -m unittest test_sim_parallel

bash serve_dp_ep.sh tp8                     # 基线
bash serve_dp_ep.sh dpep                    # -dp 8 -ep
bash serve_dp_ep.sh eplb                    # + --enable-eplb，日志里看 balancedness
bash serve_dp_ep.sh dcp                     # -tp 8 -dcp 8
bash serve_dp_ep.sh bench                   # 另开一个终端，对起好的服务压测
```

模拟器的假设：DeepSeek-V3 线性层 FP8（1 字节），embedding / lm_head / gate BF16；Llama、Qwen3 全 BF16；忽略 norm 和 FP8 scale；每卡激活 + 非 torch + CUDA Graph 合计记 6 GiB；KV cache 用 BF16。通信只算字节，时间是 NVLink 单向 450 GB/s 的带宽下限，没有算启动延迟和 RDMA。EPLB 的负载是对数正态合成数据，副本之间按均分近似哈希路由。与真机不一致时以真机日志为准。

## 期望输出

`--scenario layout`：

```
== TP2 x PP4（parallel_state.py docstring 的例子）  world=8
  tp   [[0, 1], [2, 3], [4, 5], [6, 7]]
  pp   [[0, 2, 4, 6], [1, 3, 5, 7]]

== DP2 x PP2 x TP2，MoE 模型  world=8
  tp   [[0, 1], [2, 3], [4, 5], [6, 7]]
  pp   [[0, 2], [1, 3], [4, 6], [5, 7]]
  dp   [[0, 4], [2, 6], [1, 5], [3, 7]]
  ep   [[0, 1, 4, 5], [2, 3, 6, 7]]

== DP2 x TP4 + DCP2，MoE 模型  world=8
  tp   [[0, 1, 2, 3], [4, 5, 6, 7]]
  dp   [[0, 4], [1, 5], [2, 6], [3, 7]]
  dcp  [[0, 1], [2, 3], [4, 5], [6, 7]]
  ep   [[0, 1, 2, 3, 4, 5, 6, 7]]

== TP8 x DP2（两机 16 卡，稠密模型）  world=16
  tp   [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
  dp   [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]

读法：rank = ((dp*PP + pp)*PCP + pcp)*TP + tp，TP 在最内层，所以一个 TP 组总是相邻的 GPU（同一台机器的 NVLink）。
```

`--scenario memory`：

```
== Llama-3.1-70B（70.6B，BF16）@ H100 SXM，util 0.9，每卡杂项 6 GiB
方案                      GPU     每卡权重 GiB     每卡 KV 余量 GiB    每卡每 token KV B     一个引擎 token    全部 GPU token
TP8                       8         16.4             48.9            40,960      1,280,658       1,280,658
TP4 x DP2                 8         32.9             32.4            81,920        425,020         850,041
TP2 x DP4                 8         65.7             -0.4           163,840              0               0
PP2 x TP4                 8         16.4             48.9            40,960      1,280,658       1,280,658
PP8                       8         17.9             47.4            40,960      1,242,182       1,242,182

== DeepSeek-V3（671.0B，FP8 线性层）@ H200，util 0.9，每卡杂项 6 GiB
方案                      GPU     每卡权重 GiB     每卡 KV 余量 GiB    每卡每 token KV B     一个引擎 token    全部 GPU token
TP8                       8         79.3             40.4            70,272        616,681         616,681
TP8 + EP                  8         79.3             40.4            70,272        616,681         616,681
DP8 + EP8                 8         93.9             25.7            70,272        393,297       3,146,379
DP8 + EP8 + 32 冗余         8        103.4             16.2            70,272        247,901       1,983,205
TP8 + DCP8                8         79.3             40.4             8,784      4,933,444       4,933,444
DP16 + EP16（两机）          16         55.8             63.8            70,272        974,885      15,598,154

== Qwen3-235B-A22B（235.1B，BF16）@ H200，util 0.9，每卡杂项 6 GiB
方案                      GPU     每卡权重 GiB     每卡 KV 余量 GiB    每卡每 token KV B     一个引擎 token    全部 GPU token
TP8                       8         54.9             64.7            48,128      1,443,958       1,443,958
TP8 + DCP2                8         54.9             64.7            24,064      2,887,915       2,887,915
TP4 x DP2 + EP            8         56.7             63.0            48,128      1,404,724       2,809,448
DP8 + EP8                 8         67.8             51.9           192,512        289,259       2,314,072

DP 行的 "一个引擎" 指一个 DP rank；"全部 GPU" 是所有 DP rank 加起来。MLA 在 TP 下每卡存完整 latent，8 张卡存的是同一批 token。
```

`--scenario comm`：

```
== 每个 decode step 的集合通信（一个 rank 的视角，batch = 每个 DP rank 上的 token 数）
配置                                    次数       每次消息      本卡发送 MiB      带宽下限 us    全组每 token MiB
Llama-70B TP8 all-reduce B=64        161     1024KiB         281.8          657            35.22
Llama-70B TP8 all-reduce B=256       161     4096KiB        1127.0         2626            35.22
Llama-70B PP4 send B=64                3     2048KiB           2.0            5             0.12
Llama-70B PP4 send B=256               3     8192KiB           8.0           19             0.12
DSV3 TP8 all-reduce B=64             123      896KiB         188.3          439            23.54
DSV3 TP8 all-reduce B=256            123     3584KiB         753.4         1755            23.54
DSV3 TP8 all-reduce B=512            123     7168KiB        1506.8         3511            23.54
DSV3 DP8+EP8 a2a B=64                116     2140KiB         356.3          830             5.57
DSV3 DP16+EP16 a2a B=64              116     2825KiB         470.4         1096             7.35
DSV3 DP8+EP8 a2a B=256               116     8561KiB        1425.4         3321             5.57
DSV3 DP16+EP16 a2a B=256             116    11302KiB        1881.6         4384             7.35

== DeepSeek-V3 top-8 / 256 专家：一个 token 平均要发往几个远端 rank
  EP8   命中  5.29 个 rank，其中远端  4.63；按专家发不去重则远端 7.00 份
  EP16  命中  6.52 个 rank，其中远端  6.12；按专家发不去重则远端 7.50 份
  EP32  命中  7.27 个 rank，其中远端  7.04；按专家发不去重则远端 7.75 份
  EP64  命中  7.68 个 rank，其中远端  7.56；按专家发不去重则远端 7.88 份

TP 的 all-reduce 次数和层数成正比、和 batch 无关；EP 的字节数和 batch、top-k 成正比；PP 只在段边界送一次。
"全组每 token" = 所有 rank 发送量之和 / 全组这一步处理的 token 数：TP8 的 B 是全组的 batch，DP8 的 B 是每个 rank 的 batch。
```

`--scenario dcp`：

```
模型                  TP  DCP 结果                                              每卡每 token KV B
Qwen3-235B-A22B      8    1 OK，KV 头复制 2x                                            48,128
Qwen3-235B-A22B      8    2 OK，2 张卡分摊同一条序列，KV 头复制 2x                                24,064
Qwen3-235B-A22B      8    4 ERROR exceeds the maximum supported value (2)                 -
Llama-3.1-70B        8    2 ERROR requires --tensor-parallel-size (8) > KV heads (8)                 -
Llama-3.1-70B       16    2 OK，2 张卡分摊同一条序列，KV 头复制 2x                                20,480
DeepSeek-V3          8    1 OK                                                      70,272
DeepSeek-V3          8    2 OK，2 张卡分摊同一条序列                                          35,136
DeepSeek-V3          8    4 OK，4 张卡分摊同一条序列                                          17,568
DeepSeek-V3          8    8 OK，8 张卡分摊同一条序列                                           8,784

GQA：TP 超过 KV 头数时多出来的卡只能复制 KV 头；DCP 把这份复制改成按 token 切，最大 dcp = tp // kv_heads。
MLA：每个 TP rank 本来就存完整 latent（等价于 1 个 KV 头），所以 dcp 最多到 tp。
```

`--scenario eplb`：

```
== 合成负载：4 个 MoE 层，每层 256 专家，对数正态 sigma=1.0，一个 step 32,768 次专家调用
   最热专家 / 平均 = 7.9x（第 0 层）
  EP   节点 放置                             物理专家  balancedness      最忙卡 / 平均
  16    2 linear（默认，不开 EPLB）              256         0.552          1.81
  16    2 round_robin                     256         0.499          2.00
  16    2 EPLB 0 冗余                       256         0.805          1.24
  16    2 EPLB 16 冗余（每卡 +1）               272         0.984          1.02
  16    2 EPLB 32 冗余（每卡 +2）               288         0.984          1.02

  32    4 linear（默认，不开 EPLB）              256         0.365          2.74
  32    4 round_robin                     256         0.315          3.17
  32    4 EPLB 0 冗余                       256         0.540          1.85
  32    4 EPLB 32 冗余（每卡 +1）               288         0.939          1.07
  32    4 EPLB 64 冗余（每卡 +2）               320         0.941          1.06

  64    8 linear（默认，不开 EPLB）              256         0.218          4.59
  64    8 round_robin                     256         0.203          4.94
  64    8 EPLB 0 冗余                       256         0.280          3.57
  64    8 EPLB 64 冗余（每卡 +1）               320         0.717          1.39
  64    8 EPLB 128 冗余（每卡 +2）              384         0.721          1.39

每个冗余专家多占 42 MiB（FP8），58 层每卡 +1 冗余 = 2.38 GiB 权重。
vLLM 默认 window_size=1000、step_interval=3000，计数器从 3000 - 3000//4 = 2250 起步、到 3000 触发，首次重排在启动后第 750 步；SGLang eplb_rebalance_num_iterations=1000。
```

`--scenario dp`：

```
== DP4 一个 wave：各 rank 的活在不同步数干完（dp_sync_interval=16）
  干完的步数 (40, 40, 40, 40)   wave 在第  48 步结束，各 rank dummy 步数 [8, 8, 8, 8]
  干完的步数 (40, 12, 3, 33)    wave 在第  48 步结束，各 rank dummy 步数 [8, 36, 45, 15]
  干完的步数 (97, 5, 5, 5)      wave 在第 112 步结束，各 rank dummy 步数 [15, 107, 107, 107]
  干完的步数 (1, 0, 0, 0)       wave 在第   1 步结束，各 rank dummy 步数 [0, 1, 1, 1]

== 每步 token 数的 DP 补齐（CUDA Graph 尺寸 1,2,4,8..248 步长 8，256..512 步长 16）
各 rank 实际 token            同步后 CG     各 rank 实际跑                     补齐浪费
(37, 5, 0, 64)             FULL       [64, 64, 64, 64]               59%
(120, 118, 121, 2)         FULL       [128, 128, 128, 128]           29%
(300, 12, 12, 12)          FULL       [304, 304, 304, 304]           72%
(2048, 16, 16, 16)         NONE       [2048, 16, 16, 16]              0%
(8, 8, 8, 8)               FULL       [8, 8, 8, 8]                    0%

没活的 rank 也要跑一个 1 token 的 dummy batch，因为 MoE 的 dispatch/combine 需要 EP 组里每个 rank 到场。
只要有一个 rank 走 eager（比如 2048 token 的 prefill 超过最大 capture），所有 rank 都退回 NONE，不再补齐。
稠密模型开 DP 时每个 rank 是独立引擎（EngineCoreProc），没有 wave、dummy 和补齐。
```

`--scenario pp`：

```
  层数  PP 切分（get_pp_indices）           说明
  80   4 [20, 20, 20, 20]             Llama-3.1-70B，整除
  61   4 [15, 15, 16, 15]             DeepSeek-V3：余 1 层给倒数第 2 段
  94   4 [23, 24, 24, 23]             Qwen3-235B：余 2 层给倒数第 2、3 段
  80   3 [27, 27, 26]                 余 2 层：倒数第 2、3 段，第 3 段就是第 1 段
  36   5 [7, 7, 7, 8, 7]              Qwen3-8B：余 1

== Llama-3.1-70B PP4 每段权重（BF16）：首段多 embedding，末段多 lm_head
  stage 0: 20 层，33.83 GiB
  stage 1: 20 层，31.88 GiB
  stage 2: 20 层，31.88 GiB
  stage 3: 20 层，33.83 GiB

max_concurrent_batches：同步调度 = pp；V1 runner 异步 + PP 退回 pp；MRv2 异步 = pp + 1。
batch 不够 pp 个时流水线有空泡：in-flight 1 个 batch 的 PP4，每张卡 3/4 的时间在等。
```

`--scenario sglang`：

```
== 8 x H200 跑 DeepSeek-V3：同一种部署两边怎么写
  纯 TP8              vLLM:   vllm serve DSV3 -tp 8
                     SGLang: python -m sglang.launch_server --tp 8
  注意力 DP + 专家 EP     vLLM:   vllm serve DSV3 -dp 8 -ep
                     SGLang: --tp 8 --attn-dp-size 8 --ep-size 8
  同上 + DeepEP        vLLM:   ... --all2all-backend deepep_low_latency
                     SGLang: ... --moe-a2a-backend deepep（ep_size 强制 = tp）
  EPLB + 冗余          vLLM:   --enable-eplb --eplb-config '{"num_redundant_experts":32}'
                     SGLang: --enable-eplb --ep-num-redundant-experts 32
  计算/通信 overlap      vLLM:   --enable-dbo（decode >= 32 token 才切）
                     SGLang: --enable-two-batch-overlap
  DCP                vLLM:   -tp 8 -dcp 8
                     SGLang: --tp 8 --dcp-size 8

== world size 的口径
  vLLM -tp 8 -dp 8  → 64 张卡（world_size = PP x TP x PCP，DP 另乘）
  vLLM -tp 1 -dp 8  → 8 张卡（world_size = PP x TP x PCP，DP 另乘）
  SGLang --tp 8 --attn-dp-size 8 → 8 张卡（tp_size 就是总卡数，attn_dp 在里面切）

== 注意力 DP 下的默认值
  attn_dp_size=1: SGLang chunked_prefill_size 8192 →  8192，schedule_conservativeness 1.0；vLLM 每个 DP rank 各自 max_num_batched_tokens 8192
  attn_dp_size=2: SGLang chunked_prefill_size 8192 →  4096，schedule_conservativeness 0.3；vLLM 每个 DP rank 各自 max_num_batched_tokens 8192
  attn_dp_size=4: SGLang chunked_prefill_size 8192 →  2048，schedule_conservativeness 0.3；vLLM 每个 DP rank 各自 max_num_batched_tokens 8192
  attn_dp_size=8: SGLang chunked_prefill_size 8192 →  1024，schedule_conservativeness 0.3；vLLM 每个 DP rank 各自 max_num_batched_tokens 8192

  EPLB 节奏：vLLM window 1000 / interval 3000（计数器从 2250 起步，首次在第 750 步）；SGLang 每 1000 iteration 重排一次。
  enable_dp_lm_head：SGLang 在 attn TP 组内切词表，没有 DP 时自动关掉；vLLM DP rank 各自持有完整 lm_head（tp=1 时）。
```

## 读输出时要注意

- `memory` 里 DP 行的 “全部 GPU token” 是 8 个独立 KV 池的和，单条请求最长仍受一个 DP rank 的 “一个引擎 token” 限制。
- DeepSeek-V3 的 TP8 和 TP8+EP 每卡权重相同：不开 EP 时 MoE 也按 tp×dp 切，只是切法从“每个专家切 8 份”变成“每卡 32 个完整专家”。差别在通信和 GEMM 形状，不在显存。
- `eplb` 里 64 卡 8 节点的 “EPLB 0 冗余” 只到 0.280：每卡只有 4 个槽、没有副本可分，装箱能做的有限；冗余专家才是主要手段。
- `dp` 的 wave 结束步数取决于 16 步一次的同步，最多多跑 15 个 dummy 步；dummy batch 是 1 token 的 uniform decode，代价很小但不是零。
