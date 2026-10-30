# VLLM-08 前缀缓存与混合 KV 管理

运行环境：CPU，只依赖 Python 3 标准库。基线 vllm 主干 0d6f92a，sglang 8fabf5e。复用 `../07-paged-kv/sim_block_pool.py` 里的空闲队列。

| 文件 | 内容 |
|---|---|
| `sim_prefix_cache.py` | vLLM 的链式块哈希（sha256 + pickle、NONE_HASH、extra_keys、group id 后缀）、`BlockHashToBlockMap`（不去重）、LRU 空闲队列，与 SGLang radix tree（split、插入去重、叶子堆 LRU、lock_ref、extend 后 checkpoint）放进同一个调度循环比较；混合模型部分有滑窗从右往左查找、`reachable_block_mask`、分组 padding 和 SWA 显存账 |
| `test_sim_prefix_cache.py` | 16 个单测，覆盖哈希链传播、salt / LoRA、`num_tokens - 1` 上限、重复块、设计文档 10 块例子、radix split / 去重 / 锁、滑窗例子、retention 掩码、分组 |
| `trace_prefix.md` | 前缀缓存相关代码的逐行对照表（行号对应主干） |

## 用法

```bash
python3 sim_prefix_cache.py --scenario trace     # 哈希链、重复块、文档 10 块例子（两种 free 顺序）
python3 sim_prefix_cache.py --scenario hit       # 大池子：块粒度 vs token 粒度、同批共享（约 70 s）
python3 sim_prefix_cache.py --scenario pool      # 池子从 40K 到 200K token：驱逐、抢占与命中率（约 45 s）
python3 sim_prefix_cache.py --scenario hybrid    # 滑窗命中、retention、分组 padding、SWA 显存账
python3 -m unittest test_sim_prefix_cache
```

工作负载是合成的 token 序列，不依赖真实分词器：

- **chat**：64 个用户共用 1000 token 系统提示，各聊 6 轮；每轮 prompt = 上一轮 prompt + 上一轮输出 + 新消息（均值 120），输出均值 200。
- **fewshot**：400 条请求随机套用 4 个 1234 token 模板，问题均值 80，输出均值 150。
- **random**：400 条互不相同的请求，prompt 均值 1300。

调度循环两边一样：FCFS，max_running 32，单步 prefill 预算 8192 token，块不够时从 running 队尾抢占并重算。命中率 = 首次 prefill 时命中的 token / prompt token 总数（抢占后的重算不计入分母）。`+defer` 打开 SGLang LPM 策略里的同批降级：已命中 ≤ 32 token、但和同批已收下的请求共享 ≥ 32 token 前缀的请求推到下一步。

## 期望输出

`--scenario trace`：

```
== 1. 哈希链：同样 32 个 token，bs=16 → 2 个块
  NONE_HASH = sha256(pickle('vllm-none-hash')) = 066b1e89
  无 extra      : 605fd9e3 → f1a08810
  cache_salt    : d690dc8a → c3f5e484   salt 只进块 0，块 1 因父哈希不同也变了
  LoRA(sql)     : 804b81fb → eb50ab57   LoRA 进每个块
  group 0 / 1   : …00000000 / …00000001   键长 36 字节 = 32 + 4
  改第 17 个 token：块 0 相同，块 1 不同

== 2. 重复块：bs=4，prompt ABCDEF，输出 GHI（设计文档 Duplicated blocks）
  请求 1 Time 2：块表 [0, 1, 2]，已写 KV 9 token
  请求 2 Time 0：命中 4 token（ABCDEF 的上限是 5，只够 1 块），块表 [0, 3]
  请求 2 Time 1：块表 [0, 3]；哈希 EFGH 挂着块 [1, 3]，重复 1 个
  两个请求都释放后：哈希 EFGH 仍挂着块 [1, 3]（free_blocks 只动 ref_cnt 和队列，不删哈希；多余那份等 LRU 轮到它）

== 3a. 设计文档 10 块例子，设计文档的写法（释放一律 append）
  T2 请求 0：块表 [0, 1, 2, 3, 4]
  T3 请求 1（14 token，前 10 个相同）：命中 8 token，块表 [0, 1, 5, 6]
  T4 释放请求 0：空闲队列 7-8-9-4-3-2
  T5 释放请求 1：空闲队列 7-8-9-4-3-2-6-5-1-0
  T6 请求 2（29 token，前 12 个相同）：命中 12 token，块表 [0, 1, 2, 7, 8, 9, 4, 3]，驱逐 1 个缓存块

== 3b. 设计文档 10 块例子，0d6f92a 的 free_blocks（无哈希块 prepend）
  T2 请求 0：块表 [0, 1, 2, 3, 4]
  T3 请求 1（14 token，前 10 个相同）：命中 8 token，块表 [0, 1, 5, 6]
  T4 释放请求 0：空闲队列 4-7-8-9-3-2
  T5 释放请求 1：空闲队列 6-4-7-8-9-3-2-5-1-0
  T6 请求 2（29 token，前 12 个相同）：命中 12 token，块表 [0, 1, 2, 6, 4, 7, 8, 9]，驱逐 0 个缓存块
```

`--scenario hit`：

```
== 池子足够大（400000 token 槽位），不驱逐，只看粒度与同批共享

[chat] prompt token 总数 735038
            配置 |          命中率 |   实际 prefill |        重复块/页 |           步数
    vLLM bs=16 |       93.24% |        49678 |            0 |         2757
    vLLM bs=64 |       91.91% |        59454 |            0 |         2757
  SGLang page=1 |       92.80% |        52899 |         6000 |         2773
  SGLang page=16 |       92.43% |        55630 |          372 |         2773
  SGLang page=1 +defer |       93.62% |        46899 |            0 |         2773

[fewshot] prompt token 总数 525366
            配置 |          命中率 |   实际 prefill |        重复块/页 |           步数
    vLLM bs=16 |       92.86% |        37494 |            0 |         2087
    vLLM bs=64 |       91.66% |        43830 |            0 |         2087
  SGLang page=1 |       91.37% |        45340 |         8638 |         2088
  SGLang page=16 |       91.22% |        46118 |          539 |         2088
  SGLang page=1 +defer |       93.01% |        36702 |            0 |         2088

[random] prompt token 总数 555925
            配置 |          命中率 |   实际 prefill |        重复块/页 |           步数
    vLLM bs=16 |        0.00% |       555925 |            0 |         2169
    vLLM bs=64 |        0.00% |       555925 |            0 |         2169
  SGLang page=1 |        0.00% |       555925 |            0 |         2169
  SGLang page=16 |        0.00% |       555925 |            0 |         2169
  SGLang page=1 +defer |        0.00% |       555925 |            0 |         2169
```

`--scenario pool`：

```
== chat 负载，池子从小到大（token 槽位）
            槽位 |           配置 |          命中率 |         抢占重算 |        驱逐块/页 |           步数
         40000 |   vLLM bs=16 |       55.14% |         4709 |        23064 |         3004
         40000 | SGLang page=1 |       54.88% |        38288 |       403145 |         3007
         40000 | SGLang page=16 |       54.35% |        31256 |        24822 |         3014
         60000 |   vLLM bs=16 |       62.72% |            0 |        18049 |         2757
         60000 | SGLang page=1 |       62.49% |         1917 |       290600 |         2773
         60000 | SGLang page=16 |       61.87% |         4237 |        18423 |         2773
        100000 |   vLLM bs=16 |       88.03% |            0 |         3924 |         2757
        100000 | SGLang page=1 |       88.17% |            0 |        58576 |         2773
        100000 | SGLang page=16 |       87.31% |            0 |         3884 |         2773
        200000 |   vLLM bs=16 |       93.24% |            0 |            0 |         2757
        200000 | SGLang page=1 |       92.80% |            0 |            0 |         2773
        200000 | SGLang page=16 |       92.43% |            0 |            0 |         2773
```

`--scenario hybrid`：

```
== 1. 滑窗命中（设计文档：w=4，bs=1，15 token prompt）
  缓存位置 [2, 3, 4, 5, 11, 12, 13]，max_length=14（15-1）
  从右往左：命中长度 14
  块 11-13 被驱逐后：6
  再驱逐块 5：5

== 2. 一个 32K prompt 在 SW 组里缓存几块（w=128，bs=16，对齐 16）
  retention= None（稠密）： 2048 / 2048 块进哈希表
  retention=     1024：  257 / 2048 块进哈希表
  retention=    0（默认）：    8 / 2048 块进哈希表
  full 组：2048 / 2048 块（全部可命中）

== 3. 分组与 padding（kv_cache_planning.py:522）
  gpt-oss-20b        group_size=12：sw 1 组 + 0 padding 层 (0.00%)，full 1 组 + 0 padding 层 (0.00%)
  gpt-oss-20b+eagle  group_size=13：sw 1 组 + 1 padding 层 (8.33%)，full 1 组 + 0 padding 层 (0.00%)
  gemma-3-27b        group_size=10：sw 6 组 + 8 padding 层 (15.38%)，full 1 组 + 0 padding 层 (0.00%)

== 4. 设计文档 112 token 例子（bs=16，w=32，1 full 组 + 2 sw 组）
  full 组 7 块，每个 sw 组 2 块，共 11 块

== 5. 每请求 KV 占用（bf16，bs=16），相对全部按 full attention 存
  gpt-oss-20b  L=  4096：混合     99.0 MiB，全 full    192.0 MiB，省  48.4%（页 384 KiB）
  gpt-oss-20b  L= 32768：混合    771.0 MiB，全 full   1536.0 MiB，省  49.8%（页 384 KiB）
  gpt-oss-20b  L=131072：混合   3075.0 MiB，全 full   6144.0 MiB，省  50.0%（页 384 KiB）
  gemma-3-27b  L=  4096：混合    800.0 MiB，全 full   1984.0 MiB，省  59.7%（页 1280 KiB）
  gemma-3-27b  L= 32768：混合   3040.0 MiB，全 full  15872.0 MiB，省  80.8%（页 1280 KiB）
  gemma-3-27b  L=131072：混合  10720.0 MiB，全 full  63488.0 MiB，省  83.1%（页 1280 KiB）

== 6. 同样 40 GiB，能同时放下几条 L 长请求（gpt-oss-20b）
  L= 8192：vLLM 210；ratio 0.8: 118；ratio 0.2: 177；ratio 0.05: 203
  L=32768：vLLM 53；ratio 0.8: 29；ratio 0.2: 44；ratio 0.05: 50
```

## 读法

- **哈希是链式的，一个 token 不同，后面所有块都失配。** 改第 17 个 token，块 0 不变，块 1 起全变。`cache_salt` 只进第一个块，但靠父哈希一路传到最后一块，所以租户隔离不需要每块都带 salt。LoRA 名字和路径进每一块。
- **完全相同的 prompt 也只能命中到 `num_tokens - 1`。** 最后一个 token 必须重新跑一遍前向才有 logits。块对齐的 prompt 因此会整块重算：ABCDEF（bs=4）第二次来只命中 4 个 token。
- **vLLM 不去重，SGLang 去重。** 两条同样的请求同时生成，vLLM 的哈希表里 EFGH 挂着两个块，直到 LRU 把多余那份挤掉；块表只追加，没法像 V0 那样换回去。SGLang 在 extend 后插树时发现前缀已经在树里，就把自己那份页释放掉，并把 `req_to_token` 那一行改指向树里的页。`hit` 场景里 SGLang page=1 在 chat 上释放了 6000 个重复 token，就是同批里的 64 个首轮请求各自算了一份系统提示。
- **立刻缓存满块让 vLLM 天然有“同批共享”。** vLLM 在 `allocate_slots` 里分配完就把满块写进哈希表，同一步后面的请求直接命中。SGLang 要等 extend 之后插树，同一批里的请求看不到彼此，所以它在 LPM 里专门做了 in-batch 降级（`+defer` 那一行）。打开以后 chat 命中率从 92.80% 升到 93.62%，重复归零。
- **粒度的影响比想象中小。** chat 和 fewshot 上 vLLM bs=16 和 SGLang page=1 差距在 1.5 个百分点以内；bs=64 每条请求平均多丢几十个 token，命中率掉 1.2–1.3 个百分点。真正决定命中率的是池子大小：chat 负载在 40K 槽位时只剩 55% 左右。
- **驱逐顺序：尾块先走。** 一个请求释放时块列表反转后交回空闲队列，哈希覆盖最长、最不可能被别人复用的尾块排在最前面。主干里没有哈希的块（部分块）改成 prepend，所以设计文档那个 10 块例子在主干上分到的块是 6-4-7-8-9，不再驱逐块 3。
- **滑窗反过来查。** full attention 从左往右，第一次失配就停；滑窗从右往左，凑够 `cdiv(w-1, bs)` 个连续命中块就停。混合模型先查 full 得到上限，再在这个范围里查滑窗。
- **滑窗组只缓存“能被命中的”块。** 默认 `prefix_cache_retention_interval=0`：32K prompt 在 w=128 的 SW 组里只把重放边界前 8 个块放进哈希表，full 组是 2048 块全进。
- **混合模型省的是 SW 层的 KV。** gpt-oss-20b 一半层是 w=128 的滑窗，长上下文下省接近 50%；Gemma-3-27B 52/62 层是 w=1024 的滑窗，128K 时省 83.1%，代价是 6 个 SW 组里补了 8 个 padding 层（最多浪费 15.38%）。
- **SGLang 的 SWA 池按比例切。** `swa_full_tokens_ratio` 默认 0.8，SWA 池按 full 池的 80% 开；gpt-oss-20b 下这比 vLLM 统一页池少放约 45% 的长请求，调到 0.05 才接近。比例低意味着 SWA 池小，也意味着 SWA 前缀命中更少。这里 SGLang 的每请求 SWA 占用按窗口加 128 token 懒驱逐间隔估算，只是数量级参考。
