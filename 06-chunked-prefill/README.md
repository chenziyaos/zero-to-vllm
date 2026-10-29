# VLLM-06 Chunked prefill 与异步调度

运行环境：CPU + GPU

计划内容：

- sim_chunked.py：max_num_batched_tokens 对 TTFT、ITL 的影响曲线
- sim_async.py：异步调度下 CPU 调度与 GPU 前向的重叠时间线
- gpu_budget_sweep.sh：真机扫描预算并记录 TTFT、TPOT

对应飞书文档发布后，本目录补齐代码和期望输出。
