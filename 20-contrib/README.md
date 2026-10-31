# VLLM-20 读源码与提 PR：插件系统、测试与基准

运行环境：`sim_plugins.py`、`ci_select.py`、`pr_lint.py` 和三份测试只依赖 Python 3 标准库，在 CPU 上就能跑。`plugin_demo/` 需要 vLLM（0d6f92a），`/plugins/zv_demo/scheduler_config` 的示例要 2 张 GPU。`locate.sh` 和 `contrib.sh` 要在 vllm 或 sglang 仓库根目录下执行，需要 git 和 rg。

## 文件

| 文件 | 内容 |
|---|---|
| `sim_plugins.py` | 离线推演，场景有 procs / env / endpoint / platform / failure / hooks / bench。加载逻辑照抄 `vllm/plugins/__init__.py:36-159`、`envs.py:1175-1179`、`platforms/__init__.py:240-292`，SGLang 那一侧照抄 `srt/plugins/__init__.py` 和 `hook_registry.py` |
| `ci_select.py` | 复刻 vllm-project/ci-infra（eb2d628）的 step 选择逻辑：依赖匹配、`!` 排除、yaml 自身追加、run_all、docs-only；另外复刻 SGLang 的 `register_*_ci` AST 解析、1200 加权秒的上限和 LPT 分片 |
| `pr_lint.py` | 提 PR 前自查：两边的标题正则、DCO、AI 署名、500 行 RFC 门槛、用户可见入口改动后的文档提示 |
| `test_sim_plugins.py` | 26 个回归测试，覆盖进程次数、`VLLM_PLUGINS` 的空串和空格、endpoint 四道关、平台冲突、失败传播、hook 顺序、缓存命中断崖 |
| `test_ci_select.py` | 20 个回归测试，覆盖相等与前缀匹配、排除项、yaml 只追加一次、docs-only 边界、run_all 的锚定、SGLang 注册报错 |
| `test_pr_lint.py` | 14 个回归测试，SGLang 的合法和非法样例原样搬自上游 `pr-title.test.cjs` |
| `data/ci_steps.json` | ci-infra eb2d628 解析 vllm 0d6f92a 的 `.buildkite/test_areas/*.yaml` 得到的快照：36 个 yaml，254 个 step |
| `data/vpaths.txt.gz` | vllm 0d6f92a 的 `git ls-files`（8636 个路径），用来找永远匹配不上的依赖 |
| `plugin_demo/` | 可以 `pip install -e` 的插件包：general 插件注册模型，endpoint 插件加路由，`--worker-extension-cls` 给 worker 加方法。说明见目录里的 README |
| `locate.sh` | 从日志、参数名、环境变量、行为变化定位到源码和 commit，再到测试和 CI step |
| `contrib.sh` | 从 fork 到合入的命令清单：开发环境、本地测试、A/B 基准、签名提交、两边的 CI 触发命令 |
| `contrib_matrix.md` | vLLM 与 SGLang 的插件、测试与 CI、PR 规范对照表，每一行都标了出处 |

## 用法

```bash
python3 sim_plugins.py --scenario all
python3 sim_plugins.py --scenario procs --tp 8
python3 sim_plugins.py --scenario bench --num-prompts 300
python3 ci_select.py --scenario all
python3 ci_select.py --diff vllm/plugins/__init__.py tests/plugins_tests/test_platform_plugins.py
python3 pr_lint.py --scenario demo
python3 pr_lint.py --title "[Fix] scheduler: Fix priority handling" --repo both
python3 pr_lint.py --git origin/main..HEAD
python3 -m unittest -v
bash locate.sh             # 列出所有段落
bash contrib.sh            # 列出所有段落
```

## 模拟器的假设

- **procs**：默认 spawn 且 modelinfo 缓存未命中，与 `vllm serve` 首次启动一致。`--cache-hit` 会少掉模型探测子进程那一次。SGLang 按 `engine.py:1995` 强制 spawn 计。
- **env / endpoint**：已安装的插件是自带的两个 LoRA resolver，加上 `zv_demo` 和 `zv_demo_api`。endpoint 的 503 / 500 / 200 对应 `plugin_demo/zv_demo_plugin/endpoints.py` 的三条分支，500 的报错原文取自 `serial_utils.py:485-508`。
- **failure**：只演示“谁会把服务带下去”。vLLM 的 general 插件报错用自带 LoRA resolver 没设缓存目录时的真实报错。
- **bench**：KV 容量沿用 06 章 Qwen3-8B on H100 的 24957 块（399k token），前缀缓存按 LRU 淘汰。random 数据集在同一个 seed 下生成相同的 prompt。单条 prefill 的耗时用 06 章的 prefill 耗时模型估算。
- **ci_select**：只算 NVIDIA 侧非 optional 的 step。“GPU·分钟上限”是 timeout × 卡数，实际耗时比这个低。AMD 镜像 step 单独计数，不计入上限。

## 期望输出

### procs

```
-- vLLM（spawn，modelinfo 缓存未命中）
   API server / LLM() 主进程                  执行    arg_utils.py:930
   modelinfo 检查子进程                         执行    registry.py:1636（subprocess，总是全新解释器）
   engine core                             执行    core.py:139
     worker rank 0..3                      执行    worker_base.py:289
   合计执行 7 次，分布在 7 个进程
-- SGLang（engine.py:1995 强制 spawn）
   Engine 主进程（tokenizer manager）           执行    engine.py:286
     scheduler pp0 tp0..tp3                执行    scheduler.py:6408
   detokenizer                             执行    detokenizer_manager.py:552
   合计执行 6 次，分布在 6 个进程
```

### env

```
   VLLM_PLUGINS                                    general 组                                         endpoint 组
   未设置                                             lora_filesystem_resolver, lora_hf_hub_resolver, zv_demo（无）
   ''                                              （无）                                               （无）
   'zv_demo,zv_demo_api'                           zv_demo                                           zv_demo_api
   'zv_demo, zv_demo_api'                          zv_demo                                           （无）

   SGLANG_PLUGINS                    SGLang general 组
   ''                                zv_demo, other_hooks
   'zv_demo, other_hooks'            zv_demo, other_hooks
```

### endpoint

```
   VLLM_PLUGINS 未设置                        不挂    Found endpoint plugin(s) ['zv_demo_api'] but VLLM_PLUGINS is not set.
   VLLM_PLUGINS=zv_demo,zv_demo_api        挂上
   required_tasks=generate，pooling 服务      不挂    Skipping endpoint plugin zv_demo_api: requires one of tasks ('generate',)
   render server（engine_client=None）       503  requires an engine, which this server does not have
   没配 --worker-extension-cls               500  Method 'zv_get_scheduler_config' is not implemented.
   配了 ZvWorkerExtension                    200  [{'rank': 0, 'max_num_seqs': 256}, {'rank': 1, 'max_num_seqs': 256}]
```

### platform / failure

```
   vLLM   两个 OOT 都探测成功                → RuntimeError: Only one platform plugin can be activated, but got: ['my_npu', 'my_npu_v2']
   SGLang 两个插件探测成功                    → RuntimeError: Multiple platform plugins activated: a, b. Set SGLANG_PLATFORM to select one.
   general 插件 func() 抛错                vLLM    启动失败：ValueError: VLLM_LORA_RESOLVER_CACHE_DIR must be set
   general 插件 func() 抛错                SGLang  继续启动，日志：Failed to execute general plugin: zv_demo: V
   两个插件 register_oot 同名                vLLM    启动失败：AssertionError: Duplicate op name: RMSNorm
```

### hooks

```
   注册顺序 h1:AFTER → h2:AROUND → h3:BEFORE
   执行轨迹 h3(BEFORE) → h2(AROUND 进) → 原函数 → h1(AFTER) → h2(AROUND 出)
   注册顺序 timer:AROUND → mine:REPLACE
   执行轨迹 timer(AROUND 进) → mine(REPLACE) → timer(AROUND 出)
   注册顺序 h1:AFTER → r1:REPLACE → h2:AROUND
   执行轨迹 h2(AROUND 进) → r1(REPLACE) → h1(AFTER) → h2(AROUND 出)
```

### bench

```
   KV 容量 24957 块 = 399312 token；一轮 prompt + 输出共 1152000 token
   num-prompts 300  （0.87× 容量）同 seed 复跑命中  98.4%
   num-prompts 400  （1.15× 容量）同 seed 复跑命中   0.0%
   num-prompts 1000 （2.88× 容量）同 seed 复跑命中   0.0%
   单条 prefill：冷 34.5 ms，命中 1008 token 后只算 16 个 6.3 ms
```

### ci select

```
   快照：254 个 step，82 个 optional（PR 默认不跑），21 个 AMD 原生
   改动                状态                     step 数      GPU·分钟上限  +AMD
   改插件加载器            按依赖选择                      39          2175    37
   只改插件测试            按依赖选择                       1            70     1
   改调度器              按依赖选择                      56          2886    50
   改 KV connector    按依赖选择                       0             0     2
   改 bench 工具        按依赖选择                      32          1850    30
   改 plugins.yaml    按依赖选择                       1            70     1
   改 CPU 后端 kernel   按依赖选择                       8           365     8
   改 CUDA kernel     run_all                   154          7481   155
   只改设计文档            docs-only，整条流水线跳过           0             0     0
   文档 + 一行代码         按依赖选择                      43          2290    40
```

### ci dead

```
   engine                                tests/test_config                       → tests/test_config.py
   samplers-test                         vllm/sampling_metadata.py               → （仓库中已不存在）
   共 10 条。matches() 只认“相等”或“以 dep/ 开头”，不做前缀模糊匹配，少写 .py 就永远不命中
```

### ci sglang

```
   est_time 用变量        报错：est_time is a required constant in register_cuda_ci()
   est_time=400  4-gpu-h100      1600  超限：挪去 nightly，或写 ci-cost-override: 理由
   est_time=150  8-gpu-h200      1200  通过
   最长分片 800 s，下界 max(最大单文件, 总和/3) = 800 s
```

### pr_lint demo

```
   标题                                                vLLM    SGLang
   [Bugfix][Scheduler] Fix priority handling         通过      不通过：Type 只能是 Feat, Fix, Refactor, Perf, Doc, Test, CI, Chore
   [Fix] [AMD] scheduler: Fix priority handling      通过      通过
   [Fix] [Amd] scheduler: Fix priority handling      通过      不通过：硬件标签 [Amd] 要写成 [AMD]
   Revert "[Fix] scheduler: Fix priority handling"   不通过     通过
   调度器重写             非 kernel/data/config/test 改动 690 行（总 810 行）> 500：先开 RFC issue，否则会被打 rfc-required
```
