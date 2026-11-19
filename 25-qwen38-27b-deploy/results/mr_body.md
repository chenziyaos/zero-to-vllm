## 背景

第 25 篇实验 `25-qwen38-27b-deploy` 已随 !10 合入。合入后复核发现若干事实口径需要纠正，本 MR 只做修正，不改规划器公开接口：

- 旧稿把 Qwen3.5-27B 的内容平移成 Qwen3.8；本篇统一为 Qwen/Qwen3.8-27B，实现层标识（`Qwen3_5ForConditionalGeneration`、parser `qwen3_coder`）按真实代码保留
- MTP 不是「默认 k=3」：checkpoint 只带草稿头，`num_speculative_tokens` 是 opt-in，仅 2×5090 两档命令默认给 k=3
- 接受率必须用 `/metrics` 两个计数器的前后快照差算，新增 `bench_checks.py` 落地断言
- recipe 在独立仓库 vllm-project/recipes（commit 0511d96），不在 vllm 主干；估算器 fit 场景的来源链接已更正
- 旧 GPTQ 对照结论写反：Qwen3.5-27B-GPTQ-Int4（28.16 GiB，只量化 MLP）小于 3.8 FP8（28.75 GiB），但大于四种 4 bit 档

## 改动

- `deploy_planner.py`：rank 固定 gmu 0.90 / fp8 KV / max-num-seqs 8，与 serve.sh 同口径；每行标注上下文长度，去掉「消费级唯一 verified」「必挂」这类绝对结论；TP 限制改为头数整除约束的陈述；decode all-reduce 改按每层两次小消息；ckpt 输出补充 GPTQ 对照与 vram_minimum 声明值口径
- `commands.py`：MTP 默认值改为按档位决定（TP2 两档 k=3，其余关闭），`--mtp 0` 可显式关闭
- `bench_checks.py`（新增）：冒烟响应断言（算术题必须为 323、工具调用必须解析出 get_weather）和 accepted/draft 计数器差值计算，纯标准库
- `bench.sh`：重写为 `set -euo pipefail` + tee 落盘，不 grep、不吞退出码；long 按上下文预算扣输出和模板余量；MTP 拆成 mtp-snapshot / mtp 两步
- `serve.sh`：完整日志落盘并保留失败码；h100 档 max-model-len 对齐 32768；清理无用变量和绝对化注释
- `test_deploy_planner.py`：33 → 46 个测试，新增命令生成、冒烟断言、计数器口径、recipe 来源与 rank 文案的回归
- `results/sim.txt`：用修正后的规划器重新生成
- `README.md`：用法、选卡表、口径段同步

## 测试

- [x] `python3 -m unittest test_deploy_planner`：46 passed
- [x] `bash -n serve.sh && bash -n bench.sh` 语法检查
- [ ] 真机启动 / 压测：无 GPU 环境，本次未执行；脚本不启动服务，recipe 数字只作上游记录引用
