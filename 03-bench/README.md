# VLLM-03 指标与压测

运行环境：GPU

计划内容：

- bench.sh：vllm bench serve 的固定参数组，随机数据集与 ShareGPT 两组
- sim_goodput.py：按 TTFT、TPOT SLO 从逐请求结果里算 goodput

对应飞书文档发布后，本目录补齐代码和期望输出。
