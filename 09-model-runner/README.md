# VLLM-09 Model Runner、torch.compile 与 CUDA Graph

运行环境：GPU

计划内容：

- gpu_cudagraph_modes.sh：对比 FULL_AND_PIECEWISE、PIECEWISE、NONE 与 --enforce-eager
- sim_capture_sizes.py：捕获尺寸列表与 padding 浪费估算

对应飞书文档发布后，本目录补齐代码和期望输出。
