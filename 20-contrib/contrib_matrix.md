# vLLM 与 SGLang：读源码与提 PR 的对照

基线为 vLLM 0d6f92a、SGLang 8fabf5e、vllm-project/ci-infra eb2d628。每行都标了出处，行号对应上述版本。

## 插件

| 项 | vLLM | SGLang |
|---|---|---|
| entry point 组 | `vllm.general_plugins` / `vllm.platform_plugins` / `vllm.io_processor_plugins` / `vllm.stat_logger_plugins` / `vllm.endpoint_plugins`（plugins/__init__.py:18-30） | `sglang.srt.plugins` / `sglang.srt.platforms`（srt/plugins/__init__.py:28-29） |
| 白名单变量 | `VLLM_PLUGINS`，只按 `split(",")` 切分，不做 strip；`""` 解析为 `[""]`，结果什么都不加载（envs.py:1175-1179） | `SGLANG_PLUGINS`，切分后 strip 并丢掉空串；`""` 等同于未设置（srt/plugins/__init__.py:52-55） |
| 白名单按什么匹配 | entry point 名，即等号左边（endpoint_plugins.md:101） | entry point 名 |
| general 插件抛异常 | `func()` 外面没有 try，启动直接失败（plugins/__init__.py:77-90） | 逐个包在 try 里，只打日志，服务继续启动 |
| `load()` 抛 ImportError | 打日志，跳过该插件（:36-74） | 打日志，跳过该插件 |
| 每进程只加载一次 | 模块级 `plugins_loaded` | 模块级 `_plugins_loaded` |
| 在哪些进程里执行 | API server（arg_utils.py:930）、modelinfo 子进程（registry.py:1636）、EngineCore（core.py:139）、每个 worker（worker_base.py:289） | Engine 主进程（engine.py:286）、每个 scheduler rank（scheduler.py:6408）、detokenizer（detokenizer_manager.py:552） |
| 改行为的手段 | `ModelRegistry.register_model`、`CustomOp.register_oot`（同名会触发 assert，custom_op.py:338）、`--worker-extension-cls`（worker_base.py:309-353） | hook：BEFORE / AFTER / AROUND / REPLACE。REPLACE 在最里层，越晚注册越靠外（hook_registry.py） |
| HTTP 扩展 | endpoint 插件需显式放行；`required_tasks` 控制在哪类服务上加载；路由冲突时后注册的覆盖先注册的，不报错（endpoint_plugins.md:107-134） | 无对应插件组 |
| 平台插件冲突 | 两个及以上 OOT 平台同时激活报 RuntimeError；OOT 优先于内置平台（platforms/__init__.py:240-292） | 多个平台同时激活报 RuntimeError，提示设置 `SGLANG_PLATFORM`；指定的平台没装或 activate 返回 None 也报错 |

## 测试与 CI

| 项 | vLLM | SGLang |
|---|---|---|
| 测试分配写在哪 | `.buildkite/test_areas/*.yaml`，每个 step 列出 `source_file_dependencies` | 测试文件顶部调用 `register_*_ci(est_time, stage, runner_config)`，用 AST 解析，不执行（ci_register.py） |
| 选择规则 | 改动路径与依赖项相等，或以 `dep/` 开头；`!` 前缀表示排除（buildkite_step.py:1001-1005, 1544-1561） | 按 suite 跑全部注册测试，再用 LPT 分片（`auto_partition`） |
| yaml 自身被改 | yaml 路径被追加进它每个 step 的依赖（step.py:90-106） | — |
| 全量触发 | 改动命中 `csrc/`、`setup.py`、`CMakeLists.txt`、`requirements/cuda.txt` 等（ci_config.yaml）；打 `ready-run-all-tests` 标签 | `/run-full-ci` |
| 只改文档 | 改动只涉及 docs/、recipes/、*.md、mkdocs.yaml 时，整条流水线跳过（pipeline_generator.py:486-501） | — |
| 新测试的成本约束 | 无硬性数值 | 单个 backend 的 PR 注册测试，est_time × GPU 数 ≤ 1200；超出需写 `ci-cost-override:`（test/README.md:115-129） |
| 禁用测试 | 在 yaml 中设置 `optional` 或删掉 step | `disabled="原因 #issue, until YYYY-MM-DD"` |
| 触发方式 | 加 `ready` 标签；评论 `/ci run`、`/ci retry`、`/amd-ci run`（contributing/README.md:345-365） | 评论 `/tag-and-rerun-ci`、`/run-cpu-ci`、`/rerun-failed-ci`、`/rerun-test -c` |

## PR 规范

| 项 | vLLM | SGLang |
|---|---|---|
| 标题正则 | `^(\[[^][]+\])+ +[^[:space:]].*$`，标签内容不限（pr-title.yml:34） | `[Type] scope: title` 或 `[Type] [HW] scope: title`；Type 只有 8 个；HW 必须全大写，Intel / Apple / Huawei / Ascend 例外（pr-title.yml:20-60） |
| 标题检查何时生效 | PR 非 draft 且带 `ready` 标签 | 每次编辑标题都会检查 |
| Revert | 标题必须以标签开头，所以 `Revert "..."` 不合规 | 允许 `Revert "原标题"` |
| 签名 | DCO：每个 commit 都要有 `Signed-off-by:`（README.md:190-196） | 无 DCO 要求 |
| AI 辅助 | 要求有人实际参与；PR 描述中注明；commit 加 `Co-authored-by:` 等 trailer（README.md:206-228） | — |
| 大改动 | 不含 kernel / data / config / test 超过 500 行时先开 RFC，否则会被打上 `rfc-required` 标签（README.md:309-312） | — |
| 文档 | 用户可见的改动必须在同一个 PR 里改文档（README.md:271-287） | — |
| open PR 上限 | 没有写权限的贡献者最多 6 个（README.md:385-387） | — |
| 基准复跑 | `--seed` 默认 0；`/reset_prefix_cache` 需要开启 `VLLM_SERVER_DEV_MODE=1`（envs.py:1473） | `--seed` 默认 42；只有设置 `--flush-cache` 或 `SGLANG_IS_IN_CI` 时才 flush |
