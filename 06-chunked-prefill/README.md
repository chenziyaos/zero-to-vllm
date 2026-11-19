# VLLM-06 Chunked prefill 与异步调度

运行环境：CPU，只依赖 Python 3 标准库。基线 vllm 主干 0d6f92a，sglang 8fabf5e。

| 文件 | 内容 |
|---|---|
| `sim_chunked.py` | 四种调度方式的时间线模拟：vLLM V1 默认（decode 先排、剩余预算切 prompt）、vLLM 关 chunked、SGLang 默认（prefill batch 优先）、SGLang `--enable-mixed-chunk`；同步 / 异步两种时间模型 |
| `trace_chunked.md` | chunked prefill 与异步调度相关代码的逐行对照表（行号对应主干） |

## 用法

```bash
python3 sim_chunked.py                              # 混合负载 6 req/s，vLLM 默认，预算 8192
python3 sim_chunked.py --sweep                      # 预算 512 … 16384 × 四种模式
python3 sim_chunked.py --scenario long              # 64 条 decode + 一条 32k prompt，打印尖刺那几步
python3 sim_chunked.py --scenario long --mode vllm-nochunk
python3 sim_chunked.py --scenario long --mode sglang
python3 sim_chunked.py --scenario long --sweep
python3 sim_chunked.py --scenario burst --sweep     # 2000 条同时到达：预算对吞吐的影响
python3 sim_chunked.py --threshold 2048             # long_prefill_token_threshold
python3 sim_chunked.py --sync                       # 关掉异步调度 / overlap
python3 sim_chunked.py --scenario cpu               # 纯 decode：CPU 开销与异步收益
```

时间模型（H100 SXM + Qwen3-8B BF16）：
- GPU 一步 = max(读权重 16.4 GB + 读 KV 144 KiB/token，按 3.35 TB/s；线性层 2·P·n + 注意力 FLOPs，按 989 TFLOPS × 60% × n/(n+192)) + 0.3 ms。
- CPU 一步 = 0.8 ms + 8 µs × 请求数。
- 同步一步 = CPU + GPU，异步一步 = max(CPU, GPU)。

模拟不包含抢占：KV 超出容量时只打印提示，抢占见 VLLM-05。结果只用来比较相对快慢，绝对值别拿去报数。

## 期望输出

32k prompt 插进 64 条 decode（`--scenario long`）：

```
vllm（预算 8192）
 step        时刻      本步耗时  token decode  prefill 段
  121    2.001s     7.6ms     64     64
  122    2.266s   265.7ms   8192     64  R64:8128
  123    2.599s   332.9ms   8192     64  R64:8128
  124    2.999s   400.1ms   8192     64  R64:8128
  125    3.467s   467.3ms   8192     64  R64:8128
  126    3.494s    27.8ms    320     64  R64:256
32k prompt 的 TTFT 1.50 s；短请求 ITL max 467.3 ms

vllm-nochunk
  122    3.454s  1449.8ms  32832     64  R64:32768
32k prompt 的 TTFT 1.45 s；短请求 ITL max 1.45 s

sglang（chunked_prefill_size 8192，不开 mixed）
  123    2.271s   266.1ms   8192      0  R64:8192
  ...
  126    3.479s   470.9ms   8192      0  R64:8192
32k prompt 的 TTFT 1.48 s；短请求 ITL max 1.48 s
```

`--scenario long --sweep`（vLLM 默认）：

| 预算 | 32k TTFT | 短请求 ITL max |
|---|---|---|
| 512 | 2.20 s | 39.7 ms |
| 2048 | 1.62 s | 128.6 ms |
| 8192 | 1.50 s | 467.3 ms |
| 16384 | 1.48 s | 860.2 ms |
| 关 chunked | 1.45 s | 1.45 s |

混合负载 `--sweep`（节选）：

```
    预算 模式                步数   输出 tok/s  TTFT p50  TTFT p99  ITL p50  ITL p99  ITL max
   512 vllm           12918      1,669   45.9 ms    1.28 s   6.5 ms  25.9 ms  27.4 ms
  2048 vllm           12963      1,670   34.5 ms  842.1 ms   6.4 ms  75.6 ms  84.4 ms
  8192 vllm           12977      1,670   34.3 ms  758.2 ms   6.4 ms 134.6 ms 286.9 ms
  8192 sglang         12940      1,670   33.8 ms  758.0 ms   6.4 ms  40.3 ms   1.95 s
  8192 sglang-mixed   12977      1,670   34.3 ms  758.2 ms   6.4 ms 134.6 ms 286.9 ms
     — vllm-nochunk   12984      1,670   34.5 ms  875.4 ms   6.4 ms  40.2 ms 663.2 ms
```

离线吞吐 `--scenario burst --sweep`（输出 tok/s）：

| 预算 | vllm | sglang | sglang-mixed |
|---|---|---|---|
| 512 | 3,024 | 2,658 | 3,024 |
| 2048 | 3,744 | 3,164 | 3,757 |
| 8192 | 3,534 | 3,312 | 3,563 |
| 16384 | 3,505 | 3,343 | 3,537 |

同步改异步（混合负载，预算 8192）：步数从 11,193 增加到 12,977，ITL p50 从 7.5 ms 降到 6.4 ms，ITL p99 从 180.8 ms 降到 134.6 ms。

## 读法

- **预算在 TTFT 和 ITL 之间二选一。** 32k prompt 场景里，预算从 512 加到 16384，长请求的 TTFT 从 2.20 s 降到 1.48 s；同时陪跑的 decode 单步最长从 40 ms 涨到 860 ms。关掉 chunked 相当于这条曲线的端点。
- **SGLang 默认的 prefill batch 不带 decode。** 同样切成 8192，vLLM 每一段都带着 64 个 decode；SGLang 默认的 prefill batch 一个 decode 也不带，四段连跑 1.48 s，这段时间里 decode 一个 token 都不出。开 `--enable-mixed-chunk` 后两边行为一致。
- **ITL p99 和 ITL max 可能朝相反方向变。** 混合负载下，sglang 默认的 p99 只有 40 ms，因为大多数 decode 步没有被打断；max 却到 1.95 s，因为被打断的那一次停得很久。看尾延迟要两个一起看。
- **离线吞吐在预算 2048 附近最高。** 预算太小，每步 token 少，GEMM 吃不满。预算太大，prefill 很早就做完，之后剩下大量只有 decode 的步，每步把权重整个读一遍只产出一两百个 token；这正是 Sarathi 论文里 piggyback 的论证。峰值落在哪里取决于负载里长短 prompt 的比例，真机上要自己扫。
- **异步把 CPU 那 1–5 ms 藏到 GPU 后面。** 在这个模型里收益是 14–25%。CPU 开销越重（多模态预处理、结构化输出、Python 侧采样参数多），收益越大。
