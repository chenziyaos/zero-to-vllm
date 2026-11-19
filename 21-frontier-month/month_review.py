#!/usr/bin/env python3
"""VLLM-21：一个月主干复盘的统计脚本，只用 Python 标准库。

数据在 data/ 下，由 vllm 主干 0d6f92a 和 sglang 1ab008f5 在同一窗口
（2026-09-29T09:46:03Z → 2026-10-29T09:46:04Z）的 git log 导出：
  vllm_commits.tsv.gz    sha date author ai ins del top_dirs title（author 已匿名成序号）
  sglang_commits.tsv.gz  同上，ins/del/top_dirs 未导出（blob:none 克隆，numstat 要回源）
  release_notes.json     v0.31.0 / v0.32.0 发行说明里引用的 PR 编号

用法：
  python3 month_review.py --scenario all
  python3 month_review.py --scenario themes --show 5
  python3 month_review.py --scenario coverage --split 2026-10-12
  python3 month_review.py --grep "lookahead|skip draft"
"""
import argparse
import collections
import datetime as dt
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WINDOW = ("2026-09-29T09:46:03Z", "2026-10-29T09:46:04Z")
WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# 主题桶：标题命中任一正则即归入。一个 PR 可以落进多个桶，桶之间不能相加。
THEMES = [
    ("rust", "Rust 前端", r"\[rust|rust frontend|\brust\b|pyo3"),
    ("mrv2", "Model Runner V2", r"mrv2|model runner v2|model_runner_v2|\bv2 model runner"),
    ("spec", "投机解码", r"spec(ulative)?[ _-]?dec|\beagle|\bmtp\b|dflash|dspark|ngram|draft|medusa|suffix decod|\bfly\b|xpress"),
    ("kvoff", "KV offload / 分层", r"offload|tiering|\bumbp|cpu[- ]tier|lmcache"),
    ("kvconn", "KV connector / PD", r"nixl|mooncake|kv[ _-]?connector|\bp/d\b|\bpd\b|disagg"),
    ("hisparse", "HiSparse", r"hisparse"),
    ("ds", "DeepSeek 系", r"deepseek|dsv4|dsv3|ds-?v4|deepgemm|deepep|flashmla|\bmla\b|eplb|engram"),
    ("rocm", "ROCm / AMD", r"rocm|\bamd\b|aiter|mi3\d\d|gfx9"),
    ("security", "安全", r"security|\bcve\b|ghsa|cors|ssrf|sanitiz|budget for media|media byte"),
    ("faststart", "冷启动", r"fast start|preload|weight cache|criu|snapshot|download-kernels|autotune cache"),
    ("rl", "RL / 权重更新", r"\brl\b|weight (sync|update|transfer)|update_weights|rollout"),
    ("remove", "删除 / 弃用", r"\bremove|\bdeprecat|\bdrop\b|\bdelete"),
]

# 标题里的模型名，归一后计数
MODELS = [
    ("dsv4.1", r"(deepseek[- ]?v4\.1|dsv4\.1|ds-?v4\.1)"),
    ("dsv4", r"(deepseek[- ]?v4(?!\.1)|dsv4(?!\.1)|ds-?v4(?!\.1))"),
    ("kimi-k3", r"kimi[- ]?k3"),
    ("glm5.3", r"glm[- ]?5\.3"),
    ("minimax-m3", r"minimax[- ]?m3"),
    ("qwen4", r"qwen[- ]?4"),
    ("step5", r"step[- ]?5"),
]

AI_NAMES = r"co-authored-by: *(claude|codex|openai codex|cursor|vllm-agent|gemini|copilot)"


def load(name):
    rows = []
    with gzip.open(os.path.join(DATA, name), "rt", encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            r = dict(zip(head, p))
            r["ai"] = int(r["ai"])
            r["ins"] = int(r["ins"])
            r["del"] = int(r["del"])
            r["dt"] = dt.datetime.fromisoformat(r["date"])
            m = re.search(r"\(#(\d+)\)\s*$", r["title"])
            r["pr"] = int(m.group(1)) if m else None
            rows.append(r)
    return rows


def tags(title):
    m = re.match(r"((?:\s*\[[^\]]+\])+)", title)
    return [t.strip() for t in re.findall(r"\[([^\]]+)\]", m.group(1))] if m else []


def utc_day(r):
    return r["dt"].astimezone(dt.timezone.utc).date()


def theme_hits(rows, show=0):
    out = []
    for key, name, pat in THEMES:
        rx = re.compile(pat, re.I)
        hit = [r for r in rows if rx.search(r["title"])]
        out.append((key, name, hit))
    return out


# ------------------------------------------------------------------ 场景
def scale(v, s):
    def one(rows, name):
        days = collections.Counter(utc_day(r) for r in rows)
        authors = len({r["author"] for r in rows})
        wk = sum(1 for r in rows if r["dt"].weekday() >= 5)
        prs = {r["pr"] for r in rows if r["pr"]}
        reverts = sum(1 for r in rows if r["title"].lower().startswith("revert"))
        top = ", ".join(f"{d.strftime('%m-%d')} {n}" for d, n in days.most_common(3))
        print(f"-- {name}")
        print(f"   提交 {len(rows)}，带 PR 号 {len(prs)}，作者 {authors}，日均 {len(rows) / 30:.1f}")
        print(f"   周末（作者本地时间）{wk}，占 {wk / len(rows) * 100:.1f}%；revert {reverts}")
        print(f"   最忙的三天（UTC）：{top}")
        if rows[0]["ins"] or rows[-1]["ins"]:
            ins = sum(r["ins"] for r in rows)
            dele = sum(r["del"] for r in rows)
            print(f"   增删行 +{ins} / -{dele}，净 {ins - dele:+d}")
    print(f"窗口 {WINDOW[0]} → {WINDOW[1]}（30 天）")
    one(v, "vLLM 0d6f92a")
    one(s, "SGLang 1ab008f5")


def weekday(v, s):
    for rows, name in ((v, "vLLM"), (s, "SGLang")):
        c = collections.Counter(r["dt"].weekday() for r in rows)
        mx = max(c.values())
        print(f"-- {name}（作者本地时间的星期）")
        for i, d in enumerate(WEEK):
            print(f"   {d} {c[i]:4d} " + "#" * round(c[i] / mx * 40))


def ai(v, s):
    for rows, name in ((v, "vLLM"), (s, "SGLang")):
        n = sum(r["ai"] for r in rows)
        print(f"-- {name}: 带 AI co-author 的提交 {n} / {len(rows)} = {n / len(rows) * 100:.1f}%")
        by_week = collections.defaultdict(lambda: [0, 0])
        t0 = dt.datetime.fromisoformat(WINDOW[0].replace("Z", "+00:00"))
        for r in rows:
            w = min(4, (r["dt"] - t0).days // 7)
            by_week[w][0] += r["ai"]
            by_week[w][1] += 1
        print("   按周：" + "  ".join(f"W{w + 1} {a / t * 100:.0f}%" for w, (a, t) in sorted(by_week.items())))
    print(f"   判定：commit message 里匹配 /{AI_NAMES}/i；只数署名，没署名的 AI 辅助数不到")


def tag_stats(v, s, k=12):
    for rows, name in ((v, "vLLM"), (s, "SGLang")):
        c = collections.Counter(t for r in rows for t in tags(r["title"]))
        print(f"-- {name} 标题标签 top {k}")
        print("   " + "，".join(f"[{t}] {n}" for t, n in c.most_common(k)))


def themes(v, s, show=0):
    print("主题桶按标题正则归类，一个 PR 可落入多个桶，不能相加")
    sv = {k: h for k, _, h in theme_hits(s)}
    print(f"   {'主题':<18}{'vLLM':>6}{'SGLang':>8}")
    for key, name, hit in theme_hits(v):
        print(f"   {name:<16}{len(hit):>6}{len(sv[key]):>8}")
        for r in sorted(hit, key=lambda r: -(r["ins"] + r["del"]))[:show]:
            print(f"      #{r['pr']} +{r['ins']}/-{r['del']} {r['title'][:90]}")


def models(v, s):
    for rows, name in ((v, "vLLM"), (s, "SGLang")):
        print(f"-- {name} 标题里的模型")
        out = []
        for m, pat in MODELS:
            rx = re.compile(pat, re.I)
            out.append(f"{m} {sum(1 for r in rows if rx.search(r['title']))}")
        print("   " + "，".join(out))


def churn(v, s, k=12):
    c = collections.Counter()
    for r in v:
        dirs = [d for d in r["top_dirs"].split(";") if d]
        if dirs:
            c[dirs[0]] += r["ins"] + r["del"]
    print(f"-- vLLM 按每个 PR 改动最大的目录归属的增删行 top {k}")
    tot = sum(c.values())
    for d, n in c.most_common(k):
        print(f"   {d:<28}{n:>9}  {n / tot * 100:5.1f}%")
    big = sorted(v, key=lambda r: -r["ins"])[:5]
    print("-- 新增行最多的 5 个 PR")
    for r in big:
        print(f"   #{r['pr']} +{r['ins']}/-{r['del']} {r['title'][:80]}")
    big = sorted(v, key=lambda r: -(r["del"] - r["ins"]))[:6]
    print("-- 净删除最多的 6 个 PR")
    for r in big:
        print(f"   #{r['pr']} +{r['ins']}/-{r['del']} {r['title'][:80]}")


def coverage(v, s, split="2026-10-12"):
    with open(os.path.join(DATA, "release_notes.json"), encoding="utf-8") as f:
        rn = json.load(f)
    cut = dt.date.fromisoformat(split)
    win = {r["pr"]: r for r in v if r["pr"]}
    allref = set()
    for ver, d in rn.items():
        prs = set(d["prs"])
        allref |= prs
        hit = prs & set(win)
        print(f"-- {ver}（发布 {d['published_at'][:10]}）：正文引用 {len(prs)} 个 PR，"
              f"亮点段 {len(d['highlight_prs'])} 个，其中合并在窗口内 {len(hit)} 个")
    before = [p for p, r in win.items() if utc_day(r) <= cut]
    after = [p for p, r in win.items() if utc_day(r) > cut]
    cb = sum(1 for p in before if p in allref)
    ca = sum(1 for p in after if p in allref)
    print(f"-- 窗口内 {len(win)} 个 PR，两份发行说明合计覆盖 {cb + ca} 个（{(cb + ca) / len(win) * 100:.1f}%）")
    print(f"   合并于 {split} 及以前：{len(before)} 个，覆盖 {cb}（{cb / len(before) * 100:.1f}%）")
    print(f"   合并于 {split} 以后：  {len(after)} 个，覆盖 {ca}（{ca / len(after) * 100:.1f}%）")
    print("   后半段只能读 git log 和 PR，v0.33.0 发行前没有编辑过的摘要")


def compare(v, s):
    def f(rows):
        n = len(rows)
        return dict(
            commits=n,
            authors=len({r["author"] for r in rows}),
            per_author=n / len({r["author"] for r in rows}),
            weekend=sum(1 for r in rows if r["dt"].weekday() >= 5) / n * 100,
            ai=sum(r["ai"] for r in rows) / n * 100,
            bugfix=sum(1 for r in rows if re.search(r"\[(bug ?fix|fix|hotfix)\]", r["title"], re.I)) / n * 100,
            ci=sum(1 for r in rows if re.search(r"\[ci", r["title"], re.I)) / n * 100,
        )
    a, b = f(v), f(s)
    rows = [("提交", "commits", "{:.0f}"), ("作者", "authors", "{:.0f}"), ("人均提交", "per_author", "{:.2f}"),
            ("周末占比 %", "weekend", "{:.1f}"), ("AI 署名占比 %", "ai", "{:.1f}"),
            ("修复类标签 %", "bugfix", "{:.1f}"), ("CI 标签 %", "ci", "{:.1f}")]
    print(f"   {'指标':<14}{'vLLM':>10}{'SGLang':>10}")
    for name, k, fmt in rows:
        print(f"   {name:<12}{fmt.format(a[k]):>10}{fmt.format(b[k]):>10}")


def grep(v, pat, k=40):
    rx = re.compile(pat, re.I)
    hit = [r for r in v if rx.search(r["title"])]
    print(f"{len(hit)} 条匹配 /{pat}/")
    for r in hit[:k]:
        print(f"   {r['date'][:10]} #{r['pr']} +{r['ins']}/-{r['del']} {r['title'][:100]}")


SCEN = ("scale", "weekday", "ai", "tags", "themes", "models", "churn", "coverage", "compare")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=SCEN + ("all",))
    ap.add_argument("--show", type=int, default=0, help="themes：每个桶列出改动最大的 N 个 PR")
    ap.add_argument("--split", default="2026-10-12", help="coverage：按合并日期切开的日子")
    ap.add_argument("--grep", help="按标题正则列出 vLLM 窗口内的 PR")
    a = ap.parse_args(argv)
    v, s = load("vllm_commits.tsv.gz"), load("sglang_commits.tsv.gz")
    if a.grep:
        return grep(v, a.grep)
    todo = SCEN if a.scenario == "all" else (a.scenario,)
    for sc in todo:
        print(f"== {sc}")
        if sc == "themes":
            themes(v, s, a.show)
        elif sc == "coverage":
            coverage(v, s, a.split)
        elif sc == "tags":
            tag_stats(v, s)
        else:
            globals()[sc](v, s)
        print()


if __name__ == "__main__":
    main()
