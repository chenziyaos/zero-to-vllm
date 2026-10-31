# zero-to-vllm

“从 0 开始 vLLM”飞书系列的配套代码。每章一个目录，目录名和文档章节号一一对应，索引页是 VLLM-00。

## 版本基线

- vLLM 主干提交 `0d6f92a`（2026-10-29），发行版 `v0.32.0`（2026-10-18）
- 官方镜像 `vllm/vllm-openai:v0.32.0`，CUDA 13.0；PyPI wheel 同样是 CUDA 13.0
- v0.32.0 起要求 Python 3.11 到 3.14
- CPU 练习只依赖 Python 3.11 及以上和标准库，个别脚本用 numpy

## 文件命名约定

| 前缀或后缀 | 含义 | 运行环境 |
|---|---|---|
| `sim_*.py` | 模拟器或计算器，复现主干的关键数据结构和账本 | CPU |
| `gpu_*.py` | 需要真实模型和 GPU 的实验 | GPU |
| `*.sh` | 启动、压测或部署脚本 | GPU |
| `test_*.py` | 模拟器的单元测试，`python3 -m pytest` 或直接运行 | CPU |

## 目录

| 目录 | 章节 | 内容 |
|---|---|---|
| `common/` | 全系列 | 模型结构和 GPU 规格表 |
| `01-first-server/` | VLLM-01 第一个服务 | 启动脚本、冒烟测试、日志解析 |
| `02-ledger/` | VLLM-02 推理账本 | 显存与 Roofline 计算器 |
| `03-bench/` | VLLM-03 指标与压测 | 压测脚本、goodput 统计 |
| `04-request-path/` | VLLM-04 请求旅程 | 进程拓扑与消息流 |
| `05-scheduler/` | VLLM-05 调度器 | 调度循环与抢占模拟器 |
| `06-chunked-prefill/` | VLLM-06 Chunked prefill 与异步调度 | 预算对 TTFT、ITL 的影响，异步调度时间线 |
| `07-paged-kv/` | VLLM-07 PagedAttention 与 KV 分配 | BlockPool 与块表 |
| `08-prefix-caching/` | VLLM-08 前缀缓存与混合 KV 管理 | 块哈希、驱逐、混合 KV 组 |
| `09-model-runner/` | VLLM-09 Model Runner、torch.compile 与 CUDA Graph | CUDA Graph 档位对比 |
| `10-attention-backend/` | VLLM-10 注意力后端 | 后端选择表与微基准 |
| `11-parallelism/` | VLLM-11 并行 | 并行方案账本 |
| `12-quantization/` | VLLM-12 量化 | 量化显存与精度对比 |
| `13-speculative/` | VLLM-13 投机解码 | 接受率与加速比 |
| `14-kv-offload/` | VLLM-14 KV 卸载与连接器 | 两级缓存命中与带宽账本 |
| `15-pd-disagg/` | VLLM-15 PD 分离 | P、D 配比与 KV 传输账本 |
| `16-structured-lora/` | VLLM-16 结构化输出与多 LoRA | token mask 与多 LoRA |
| `17-tuning/` | VLLM-17 调参手册 | 调参记录模板与扫描脚本 |
| `18-observability/` | VLLM-18 可观测与 profiling | 指标抓取与 trace 分析 |
| `19-cluster-rl/` | VLLM-19 集群与 RL | 路由策略模拟、sleep mode |
| `20-contrib/` | VLLM-20 读源码与提 PR | 插件加载模拟、CI 选择复刻、PR 自查、可安装的插件示例 |
| `21-frontier-month/` | VLLM-21 一个月主干复盘 | 30 天提交快照、主题分桶、发行说明覆盖率、标题检索 |
| `22-deepseek-impact/` | VLLM-22 DeepSeek 开源对 vLLM 的影响 | KV 字节账、EPLB 移植、有界重放与自适应验证模拟、组件接入对照 |

22 章已全部发布，每个目录的 README 里都写了运行环境、用法和期望输出。

## 快速试跑

```bash
cd zero-to-vllm/02-ledger
python3 sim_ledger.py --model qwen3-8b --gpu h100-sxm
python3 sim_ledger.py --model llama3.1-70b --gpu h100-sxm --tp 4 --kv-dtype fp8
```

起服务（需要 GPU）：

```bash
cd zero-to-vllm/01-first-server
bash serve.sh                     # 默认 Qwen/Qwen3-8B，端口 8000
python3 smoke_test.py
```

## 参考

- vLLM：https://github.com/vllm-project/vllm
- nano-vllm：https://github.com/GeeeekExplorer/nano-vllm
- 官方文档：https://docs.vllm.ai/
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- 同系列 SGLang 对照代码：`../zero-to-sglang`
