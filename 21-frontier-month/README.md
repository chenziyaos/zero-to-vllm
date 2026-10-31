# VLLM-21 前沿：一个月主干复盘

运行环境：`month_review.py` 和 `test_month_review.py` 只依赖 Python 3 标准库，读 `data/` 下的快照，不联网、不需要 git 和 GPU。`export_snapshot.py` 用来重新导出快照，需要本地的 vllm 和 sglang 克隆以及 git。

## 文件

| 文件 | 内容 |
|---|---|
| `month_review.py` | 复盘统计。场景有 scale / weekday / ai / tags / themes / models / churn / coverage / compare，另有 `--grep` 按标题正则列出窗口内的 PR |
| `test_month_review.py` | 17 个回归测试。正文里引用的计数都钉在这里，快照一换，测试会先失败 |
| `export_snapshot.py` | 从 git log 和发行说明 JSON 重新生成 `data/`。作者邮箱只取 sha1 后编成序号，不写进快照 |
| `data/vllm_commits.tsv.gz` | vllm 主干 0d6f92a 在窗口内的 1942 个提交：sha、日期（带作者时区）、作者序号、是否有 AI co-author、增删行、改动最多的 3 个目录、标题 |
| `data/sglang_commits.tsv.gz` | sglang 1ab008f5 在同一窗口内的 2135 个提交，字段同上，增删行和目录没有导出（blob:none 克隆，numstat 要逐个回源取 blob） |
| `data/release_notes.json` | v0.31.0、v0.32.0 发行说明引用的 PR 编号、亮点段 PR 编号和章节名 |

窗口：2026-09-29T09:46:03Z → 2026-10-29T09:46:04Z，30 天。

## 用法

```bash
python3 month_review.py --scenario all
python3 month_review.py --scenario themes --show 5          # 每个主题桶列出改动最大的 5 个 PR
python3 month_review.py --scenario coverage --split 2026-10-05
python3 month_review.py --grep "lookahead|skip draft|sync_load|budget"
python3 -m unittest test_month_review -v

# 换窗口或换 commit 时重新导出
python3 export_snapshot.py --vllm ~/src/vllm --vllm-ref 0d6f92a \
    --sglang ~/src/sglang --sglang-ref 1ab008f5 --sglang-no-numstat \
    --notes rel_v0.31.0.json rel_v0.32.0.json
```

## 口径

- **周末**按作者本地时间算（`git log --date=iso-strict` 带时区），**最忙的三天**按 UTC 日期算，两者不是一个口径。
- **AI 署名**只数 commit message 里的 `Co-authored-by: Claude / Codex / Cursor / vllm-agent / Gemini / Copilot`。没署名的 AI 辅助数不到，所以这是下限。`Signed-off-by: Claude Shannon` 这类不算。
- **主题桶**按标题正则归类，一个 PR 可以同时落进几个桶（#61482 同时属于 DeepSeek 和 ROCm），桶之间不能相加。正则写在 `THEMES` 里，改正则就会改计数。
- **模型计数**里 `dsv4` 用负向前瞻排除了 `4.1`，两个数可以相加。
- **churn** 把每个 PR 的增删行整笔记到它改动最多的那个目录上（取前两级路径），是粗略的“代码重心”，不是精确的逐目录统计。
- **coverage** 只看两份已发布的发行说明。窗口末尾已经有 v0.33.0rc1，但没有发布页，所以 10-12 以后合并的 PR 几乎都不在任何发行说明里。
- vllm 主干是 squash merge，每个提交对应一个 PR；SGLang 有 1 个提交没有 PR 号。

## 期望输出

### scale

```
窗口 2026-09-29T09:46:03Z → 2026-10-29T09:46:04Z（30 天）
-- vLLM 0d6f92a
   提交 1942，带 PR 号 1942，作者 658，日均 64.7
   周末（作者本地时间）242，占 12.5%；revert 5
   最忙的三天（UTC）：10-27 102, 10-26 99, 10-08 91
   增删行 +601102 / -128748，净 +472354
-- SGLang 1ab008f5
   提交 2135，带 PR 号 2134，作者 413，日均 71.2
   周末（作者本地时间）444，占 20.8%；revert 5
   最忙的三天（UTC）：10-09 107, 10-20 97, 10-23 96
```

### ai

```
-- vLLM: 带 AI co-author 的提交 1109 / 1942 = 57.1%
   按周：W1 45%  W2 47%  W3 56%  W4 72%  W5 70%
-- SGLang: 带 AI co-author 的提交 457 / 2135 = 21.4%
   按周：W1 22%  W2 21%  W3 21%  W4 24%  W5 15%
```

W5 只有最后两天，样本少。

### tags

```
-- vLLM 标题标签 top 12
   [Bugfix] 646，[ROCm] 250，[CI] 231，[Perf] 184，[Frontend] 135，[Rust Frontend] 115，[Model] 88，[XPU] 64，[Spec Decode] 56，[KV Offload] 52，[Core] 52，[Doc] 50
-- SGLang 标题标签 top 12
   [Fix] 460，[Perf] 296，[AMD] 238，[diffusion] 169，[CI] 154，[Feat] 152，[Refactor] 122，[Doc] 81，[NPU] 74，[Test] 35，[PD] 35，[NVIDIA] 34
```

### themes

```
主题桶按标题正则归类，一个 PR 可落入多个桶，不能相加
   主题                  vLLM  SGLang
   Rust 前端            123      64
   Model Runner V2     52       0
   投机解码               155     154
   KV offload / 分层     72      12
   KV connector / PD    92     135
   HiSparse            32      10
   DeepSeek 系         208     294
   ROCm / AMD         286     259
   安全                  18       1
   冷启动                 26       9
   RL / 权重更新           26       5
   删除 / 弃用             79      83
```

### models

```
-- vLLM 标题里的模型
   dsv4.1 68，dsv4 30，kimi-k3 56，glm5.3 35，minimax-m3 41，qwen4 26，step5 6
-- SGLang 标题里的模型
   dsv4.1 100，dsv4 127，kimi-k3 27，glm5.3 29，minimax-m3 18，qwen4 4，step5 1
```

### churn

```
-- vLLM 按每个 PR 改动最大的目录归属的增删行 top 12
   vllm/models                    172591   23.6%
   tests/v1                        93990   12.9%
   vllm/model_executor             58710    8.0%
   rust/src                        58529    8.0%
   recipes/models                  52870    7.2%
   vllm/v1                         36158    5.0%
   ...
-- 净删除最多的 6 个 PR
   #60517 +32/-4797 [Compile] Remove torch.compile-based sequence parallelism and AsyncTP (#60517)
   #62186 +0/-3093 Revert "[Docs] Add train-inference consistency agent skill" (#61551) (#62186)
   #64287 +2/-2928 [Model] Remove Keye-VL and Keye-VL-1.5 (#64287)
   ...
```

### coverage

```
-- v0.31.0（发布 2026-10-05）：正文引用 510 个 PR，亮点段 57 个，其中合并在窗口内 22 个
-- v0.32.0（发布 2026-10-18）：正文引用 330 个 PR，亮点段 49 个，其中合并在窗口内 311 个
-- 窗口内 1942 个 PR，两份发行说明合计覆盖 333 个（17.1%）
   合并于 2026-10-12 及以前：780 个，覆盖 323（41.4%）
   合并于 2026-10-12 以后：  1162 个，覆盖 10（0.9%）
```

### compare

```
   指标                  vLLM    SGLang
   提交                1942      2135
   作者                 658       413
   人均提交              2.95      5.17
   周末占比 %            12.5      20.8
   AI 署名占比 %         57.1      21.4
   修复类标签 %           34.1      22.2
   CI 标签 %           12.8       7.4
```
