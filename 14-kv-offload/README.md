# VLLM-14 KV 卸载与连接器：CPU 卸载、LMCache 与 Mooncake

运行环境：模拟器只需要 CPU 和 Python 3 标准库。`serve_offload.sh` 需要一张 H100（`canonical` 要 4 张）、装好的 vLLM，以及至少 256 GiB 主机内存；`lmcache` 要另装 LMCache 并先起 `lmcache server`；`mooncake` 要另装 Mooncake 并起 `mooncake_master`；SGLang 对照要另装 sglang。源码基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_offload.py` | 七个场景。capacity：GPU、CPU、NVMe 三级各放得下多少 token 和多少个 agentic 会话，以及两边容量参数的单位差别。ledger：回迁和重算每 token 各花多少时间，80K 前缀一轮请求的 TTFT 账。layout：旧布局和新布局下的物理块大小，与 vLLM 博客表格核对。tiers：并发会话数 × CPU 容量 → GPU 命中、CPU 命中、重算。eviction：混入一次性长文档后 LRU、ARC、store_threshold=2 的差别。lifecycle：逐行移植连接器判定逻辑，一个请求走完调度器的每一步。sglang：HiCache 的容量、写策略、三道阈值、预取超时 |
| `test_sim_offload.py` | 14 个单测：MultiConnector 的首个命中与 None 规则；在途传输返回 None；同步预算至少留 1 个 token；整段命中回退 1 个 token；不满块尾巴的处理；物理块大小与博客表格一致；每 token KV 字节；HiCache 预取超时的代码默认（1 + 0.25·tok/1024，无上限）与设计文档（封顶 30 s）两套公式；ARC ghost 调整和抗扫描；store_threshold 跳过一次性 chunk；CPU 层小于 GPU 池时 CPU 命中约为 0 |
| `serve_offload.sh` | 真机实验：base / native / arc / sync / small / simple / tiered / canonical / lmcache / mooncake 十种起法，`prefix_repetition` 压测，以及 SGLang 五种对照起法 |
| `offload_matrix.md` | 入口参数、连接器 API、调度器、OffloadingConnector、淘汰与写入、`kv_connector_extra_config` 各部分的源码行号，以及与 SGLang HiCache 的参数对照 |

## 用法

```bash
python3 sim_offload.py --scenario capacity    # 每级能放多少 token、多少个会话
python3 sim_offload.py --scenario ledger      # 回迁 vs 重算
python3 sim_offload.py --scenario layout      # 旧布局 vs 新布局的物理块大小
python3 sim_offload.py --scenario tiers       # 会话数 × CPU 容量 → 分级命中（约 8 秒）
python3 sim_offload.py --scenario eviction    # LRU / ARC / store_threshold
python3 sim_offload.py --scenario lifecycle   # 连接器生命周期
python3 sim_offload.py --scenario sglang      # HiCache 对照

python3 -m unittest test_sim_offload          # 14 个单测

bash serve_offload.sh base      # 终端 1：起服务
bash serve_offload.sh bench 64  # 终端 2：64 个 16K 前缀 × 10 轮，前缀总量约 1M token，超过 GPU 池
bash serve_offload.sh native 128 && bash serve_offload.sh bench 64   # 换 128 GiB CPU 层再压一次
bash serve_offload.sh small     && bash serve_offload.sh bench 64   # 32 GiB：外部命中率应接近 0
grep -h "External prefix cache hit rate" results/*.log | tail
```

压测的前缀总量要超过 GPU 池才能看出 CPU 层的作用：Llama-3.1-8B 在 H100 上 GPU 池约 457K token，64 个 16K 前缀约 1M token，CPU 层要大于 GPU 池（约 56 GiB）。

## 模拟器的假设

- **GPU 池**：沿用 02-ledger 的账本，H100 80GB、util 0.92，Llama-3.1-8B 单卡 457,696 token、55.87 GiB。
- **带宽**：GPU↔CPU 取 vLLM 博客实测的 DMA 单向约 50 GB/s。本地 NVMe 取 12 GB/s，RDMA 取 45 GB/s，这两个是假设值。
- **重算**：按 prefill 的 FLOP 算，线性层 2·P·n，注意力 4·L·H·d·(n·p + n²/2)，MFU 取 0.5。
- **tiers / eviction 粒度**：用 512 token 的 chunk 做前缀哈希，只判断能否命中，不模拟传输排队。回迁时间按带宽串行累加，不与计算重叠，所以偏保守。
- **ARC**：按 `v1/kv_offload/cpu/policies/arc.py` 的语义移植，不是逐行移植。LRU 是 chunk 粒度，源码是请求粒度。
- **store_threshold**：模拟器在"提交存储"时计数，源码在 lookup 时计数，两者在多轮负载下等价（每轮先 lookup 再 store）。
- **多轮负载**：按 vLLM × Mooncake 博客的 agentic 统计，首轮 12K、每轮 +2,242、30 轮，会话随机交错。
- **lifecycle**：`offload_matched`、`schedule_external`、`after_recv`、`multi_connector_pick` 逐行移植判定逻辑，不含真实的块分配。

与真机结果不一致时，以真机日志为准。

## 期望输出

`sim_offload.py --scenario tiers`（节选）：

```
    会话  CPU GiB CPU/GPU   GPU 命中   CPU 命中     重算  每轮 prefill     相对无卸载
    32        0    0.00    27.6%     0.0%  72.4%      2053ms     1.00x
    32       32    0.57    27.6%     0.0%  72.4%      2053ms     1.00x
    32      128    2.29    27.6%    35.3%  37.1%      1235ms     1.66x
    32      256    4.58    27.6%    66.2%   6.2%       299ms     6.86x
   128      512    9.16     7.4%    58.2%  34.3%      1169ms     2.05x
```

`sim_offload.py --scenario eviction`（节选）：

```
     长文档占比 策略                         GPU 命中   CPU 命中     重算  写 CPU GiB/轮
       1/2 lru                          8.3%    10.7%  81.0%         3.81
       1/2 arc                          8.3%    15.7%  76.0%         3.58
       1/2 lru+store_threshold=2        8.3%    27.1%  64.6%         0.95
```

`sim_offload.py --scenario ledger`（节选）：

```
      前缀 p    GPU 命中    CPU 回迁    NVMe 回迁      全部重算   重算/CPU
    32,000     151ms     235ms      501ms    1730ms     7.4x
    80,000     265ms     475ms     1139ms    6247ms    13.1x
```

`sim_offload.py --scenario layout`（节选）：

```
  模型                          旧：一段      新：一块    段数   博客表格（旧 / 新）
  Llama-3.1-8B                32KB    2.00MB    64   32 KB / 2 MB
  Llama-3.1-70B（TP=4）          8KB    1.25MB   160   8 KB / 1.25 MB
```
