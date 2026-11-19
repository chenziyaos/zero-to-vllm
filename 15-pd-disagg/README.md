# VLLM-15 PD 分离：NIXL、KV 传输与 P/D 感知路由

运行环境：模拟器只需要 CPU 和 Python 3 标准库。`serve_pd_nixl.sh` 需要一台两卡机器（L40S、A100、H100 都行，默认模型 Qwen2.5-7B-Instruct），装好 vLLM 和 NIXL（`pip install nixl`），并把 vLLM 源码克隆到 `$VLLM_SRC`（proxy 脚本在源码里）；SGLang 对照要另装 sglang 和 sglang-router。源码基线：vllm 主干 0d6f92a，sglang 主干 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_pd.py` | 九个场景。transfer：每 token KV、各种链路上搬一个请求要多久、按 TP 切开后每 rank 的字节数和描述符数。interference：两张 L40S 上同机 DP=2 与 1P1D 的 ITL / TTFT（传输 1.3 s 与修好后两档）。goodput：TTFT ≤ 2 s 且请求内 p99 ITL ≤ 50 ms 的达标率。ratio：五种负载下的 P:D 配比和 P→D 带宽。pushpull：pull 与 push 的关键路径，按博客的一个点反推每描述符成本。lease：租约 + 心跳 vs 旧的 480 s 硬超时。lifecycle：逐行移植调度器判定，一个请求在 P、D 上走完每一步，含 backoff 与双向第二轮。tpmap：异构 TP 时 D 的每个 rank 读 P 的哪些 rank。sglang：bootstrap、D 侧准入公式、超时与探活 |
| `test_sim_pd.py` | 20 个单测：每 token KV 与拆分指南一致（56 / 256 / 320 KiB）；描述符 = 层 × K/V × 块；D 侧整段读、本地前缀只读差额；P 侧第一轮本地算；双向阈值 64 的边界；坐标不全不读；收齐回退 1 个 token；四种 backoff；`request_finished` 五个分支；租约默认值、续期中读完、D 挂掉后最晚释放时间；TP 相等 / P 大去重 / D 大头偏移 / MLA；SGLang 准入与 DP 取模；P/D 消掉 ITL 尾巴 |
| `serve_pd_nixl.sh` | 真机实验：同机基线、pull、push、双向、短租约五种起法，toy proxy / push proxy / multiturn proxy，random 8k/256 压测与速率扫描，多轮压测，以及 SGLang NIXL 对照 |
| `pd_matrix.md` | 入口参数、调度侧、worker 侧、`kv_connector_extra_config`、proxy 的源码行号，以及与 SGLang PD 分离的参数对照 |

## 用法

```bash
python3 sim_pd.py --scenario transfer       # KV 账与链路
python3 sim_pd.py --scenario interference   # 同机 vs 1P1D 的 ITL 尾巴（约 0.5 秒）
python3 sim_pd.py --scenario goodput        # SLO 达标率（约 1 秒）
python3 sim_pd.py --scenario ratio          # P:D 配比
python3 sim_pd.py --scenario pushpull       # pull vs push
python3 sim_pd.py --scenario lease          # 租约与心跳
python3 sim_pd.py --scenario lifecycle      # 调度器判定逐步走
python3 sim_pd.py --scenario tpmap          # 异构 TP 映射
python3 sim_pd.py --scenario sglang         # SGLang 对照

python3 -m unittest test_sim_pd             # 20 个单测

# 单机 1P1D：四个终端
bash serve_pd_nixl.sh p          # 终端 1：GPU0，:8100，side channel 5600
bash serve_pd_nixl.sh d          # 终端 2：GPU1，:8200，side channel 5601
bash serve_pd_nixl.sh proxy      # 终端 3：:8192
bash serve_pd_nixl.sh sweep      # 终端 4：0.2 → 2.0 req/s，各 100 条 8k/256
grep -h "KV Transfer metrics" results/pull_d.log | tail -3
grep -h "Transfer plan" results/pull_d.log | head -1
```

换 push：终端 1–3 改成 `push_p`、`push_d`、`push_proxy`；换双向：`bidir_p`、`bidir_d`、`bidir_proxy`，压测改 `multiturn`。看租约：用 `lease 10` 起 P，发一批请求后在 P 算完时 kill D，P 的日志里 10 秒内应出现 `Releasing expired KV blocks`，`/metrics` 里 `vllm:nixl_num_kv_expired_reqs` 增加。

单机两卡之间默认 `UCX_TLS=cuda_ipc,cuda_copy,tcp`，卡间走 CUDA IPC，比跨机 RDMA 快得多。要看到传输真正成为瓶颈，把 `UCX_TLS=tcp` 强制走 TCP，或者跨两台机器、各用一张网卡。

## 模拟器的假设

- **KV**：BF16，`2 · 层 · KV头 · head_dim · 2 B`；DeepSeek-V3 按 MLA latent 每层一份。
- **描述符**：每层 K、V 各一个 region，每块一个描述符，块大小 16；push 合并相邻块的效果不计。
- **单步时间**：max(算力时间, 读权重 + 读 KV 时间) + 固定开销 3 ms；L40S 按 362 TFLOP/s BF16、864 GB/s，MFU 0.5。
- **interference / goodput**：同机是两张卡各一个实例、轮询分发，chunked prefill 预算 2048 token；1P1D 的 P 只做 prefill，算完加上传输时间 KV 到 D，D 从 `num_computed = n − 1` 开始。不模拟 D 的 KV 池溢出和抢占。
- **传输两档**：1.3 s 取自 vLLM 拆分指南在 L40S 上的实测（470 MB）；"修好"按 12 GB/s 主机中转加 20 ms 握手。
- **ratio**：P 的吞吐按 prefill FLOP / (峰值 · MFU) 再乘 0.7 的利用率；D 的并发取 ITL SLO 与 KV 池两个约束里较小的那个。
- **pushpull**：博客没有给每描述符成本，用"8 QPS × 16k 时 pull TPOT 10.4 ms、push 7 ms"这一点反推出 312 ns，只用来看量级和走向。
- **lease**：D 活着时每 5 s 发一次心跳，每次 expiry = max(旧, now + 20)；P 在 expiry 时刻释放。不计通知延迟。
- **lifecycle / tpmap / sglang**：逐行移植判定逻辑（`pull_matched`、`request_finished`、`after_recv`、`tp_sources`、`head_offset`、`sgl_admission`），不含真实的块分配和网络。

与真机结果不一致时，以真机日志为准。

## 期望输出

`sim_pd.py --scenario interference`（节选）：

```
  req/s |          同机 DP=2           |       1P1D 传输 1.3 s        |         1P1D 传输修好
        |   ITL50   ITL99   TTFT50  |   ITL50   ITL99   TTFT50  |   ITL50   ITL99   TTFT50
    0.2 |   21.1ms    22ms   0.77s  |   21.7ms    23ms   2.10s  |   21.7ms    23ms   0.86s
    0.4 |   21.7ms   189ms   0.80s  |   22.2ms    24ms   2.11s  |   22.2ms    24ms   0.87s
    2.0 |   34.4ms   207ms   1.08s  |   25.0ms    26ms  17.20s  |   25.0ms    26ms  15.96s
```

`sim_pd.py --scenario goodput`（节选）：

```
  req/s    同机 DP=2  1P1D 1.3 s    1P1D 修好
    0.2        76%          0%        99%
    0.8         7%          0%        90%
    1.5         2%          0%        37%
```

`sim_pd.py --scenario lease`（节选）：

```
  情形                                      新：释放于      旧：释放于
  D 排队 200 s 后读完（心跳续着）                   200.0s     200.0s
  D 在 3 s 时挂掉                             30.0s     480.0s
  D 在 47 s 时挂掉                            65.0s     480.0s
```

`sim_pd.py --scenario tpmap`（节选）：

```
  Qwen3-32B   P TP8 → D TP4: D0←P0,1  D1←P2,3  D2←P4,5  D3←P6,7
  Qwen3-32B   P TP4 → D TP8: D0←P0  D1←P0(+1)  D2←P1  D3←P1(+1)  ...
  Qwen2.5-7B  P TP8 → D TP2: D0←P0,2  D1←P4,6
```
