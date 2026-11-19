# zv_demo_plugin

一个最小的 vLLM 插件包，包含两个 entry point：一个 general 插件（注册模型架构，可选替换 custom op），一个 endpoint 插件（加一条 HTTP 路由，经 `collective_rpc` 读取每个 worker 的调度配置）。基线是 vllm 0d6f92a。

| 文件 | 作用 |
|---|---|
| `setup.py` | 声明两个 entry point：`vllm.general_plugins: zv_demo`、`vllm.endpoint_plugins: zv_demo_api` |
| `zv_demo_plugin/__init__.py` | `register()`：幂等、开销小、自己处理异常 |
| `zv_demo_plugin/worker_ext.py` | `--worker-extension-cls` 用的混入类，提供 `zv_get_scheduler_config` |
| `zv_demo_plugin/endpoints.py` | `GET /plugins/zv_demo/scheduler_config` |

## 安装与启动

```bash
pip install -e plugin_demo
# 一旦设置 VLLM_PLUGINS，就只加载列表里的插件；内置的两个 LoRA resolver 要保留的话，需要一起写进来
# 逗号后面不要加空格：envs.py 只做 split(",")，不做 strip，加了空格名字就对不上
export VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver,zv_demo,zv_demo_api
VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3-0.6B -tp 2 \
  --worker-extension-cls zv_demo_plugin.worker_ext.ZvWorkerExtension
curl -s localhost:8000/plugins/zv_demo/scheduler_config
```

期望输出（TP=2，每个 worker 返回一项）：

```json
{"workers": [{"rank": 0, "max_num_seqs": 256, ...}, {"rank": 1, "max_num_seqs": 256, ...}]}
```

日志里 `zv_demo: register() pid=…` 会出现多次，每个进程一次：API server、模型探测子进程、EngineCore，再加每个 worker 各一次。

## 三种失败

| 现象 | 原因 |
|---|---|
| 路由 404，日志有 `VLLM_PLUGINS is not set` 相关警告 | endpoint 插件只认显式放行，没设置 `VLLM_PLUGINS` 时不会加载 |
| 500 `Method 'zv_get_scheduler_config' is not implemented.` | 启动时没带 `--worker-extension-cls` |
| 503 | 当前是只做 CPU 渲染的 render server，`engine_client` 为 None |

设置 `ZV_DEMO_OOT=1` 后，`register()` 会用 `CustomOp.register_oot(name="SiluAndMul")` 替换激活函数。如果同名名字已经被别的插件注册，代码直接跳过，不会撞上 `custom_op.py:338` 的 `assert`。

## SGLang 侧对照

SGLang 的插件 entry point 组是 `sglang.srt.plugins`，用 `SGLANG_PLUGINS` 过滤（会 strip，空串等于没设置），由 `load_plugins` 逐个加载。每个插件都包在 try 里，失败只打日志，加载完后统一执行 `apply_hooks`。SGLang 改行为靠 hook（BEFORE / AFTER / AROUND / REPLACE），不靠混入 worker 类。详见上一级目录 `sim_plugins.py --scenario hooks`。
