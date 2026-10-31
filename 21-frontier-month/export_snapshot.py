#!/usr/bin/env python3
"""重新导出 data/ 快照：作者邮箱只取 sha1 后编成序号，不落盘。

  python3 export_snapshot.py --vllm ~/src/vllm --vllm-ref 0d6f92a \
      --sglang ~/src/sglang --sglang-ref 1ab008f5 \
      --notes rel_v0.31.0.json rel_v0.32.0.json

发行说明 JSON 用 `gh api repos/vllm-project/vllm/releases/tags/v0.32.0 > rel_v0.32.0.json` 取。
SGLang 如果是 --filter=blob:none 的克隆，加 --sglang-no-numstat，否则 numstat 会逐个回源取 blob。
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SINCE, UNTIL = "2026-09-29T09:46:03Z", "2026-10-29T09:46:04Z"
AI = re.compile(r"co-authored-by: *(claude|codex|openai codex|cursor|vllm-agent|gemini|copilot)", re.I)


def dump(repo, ref, out, numstat):
    fmt = "%x1e%H%x1f%ae%x1f%ad%x1f%s%x1f%b%x1d"
    args = ["git", "-C", repo, "log", "--date=iso-strict", "--format=" + fmt,
            f"--since={SINCE}", f"--until={UNTIL}", ref]
    if numstat:
        args.insert(4, "--numstat")
    text = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    authors, rows = {}, []
    for rec in text.split("\x1e")[1:]:
        head, _, tail = rec.partition("\x1d")
        h, ae, ad, s, b = head.split("\x1f")
        aid = authors.setdefault(hashlib.sha1(ae.lower().encode()).hexdigest(), len(authors))
        ins = dele = 0
        dirs = collections.Counter()
        for line in tail.strip().splitlines():
            p = line.split("\t")
            if len(p) == 3:
                a = int(p[0]) if p[0] != "-" else 0
                d = int(p[1]) if p[1] != "-" else 0
                ins += a
                dele += d
                parts = p[2].split("/")
                dirs["/".join(parts[:2]) if len(parts) > 2 else parts[0]] += a + d
        top = ";".join(k for k, _ in dirs.most_common(3))
        rows.append([h[:10], ad, aid, int(bool(AI.search(b))), ins, dele, top, s])
    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.write("sha\tdate\tauthor\tai\tins\tdel\ttop_dirs\ttitle\n")
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
    print(os.path.basename(out), len(rows), "commits,", len(authors), "authors")


def notes(paths, out):
    res = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        body = d["body"]
        hl = body.split("## Release Artifacts")[0]
        res[d["tag_name"]] = {
            "published_at": d["published_at"],
            "prs": sorted({int(x) for x in re.findall(r"#(\d+)", body)}),
            "highlight_prs": sorted({int(x) for x in re.findall(r"#(\d+)", hl)}),
            "sections": [l.strip("# \r") for l in body.splitlines() if l.startswith("## ")],
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(os.path.basename(out), ", ".join(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--vllm-ref", default="0d6f92a")
    ap.add_argument("--sglang", required=True)
    ap.add_argument("--sglang-ref", default="1ab008f5")
    ap.add_argument("--sglang-no-numstat", action="store_true")
    ap.add_argument("--notes", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dump(a.vllm, a.vllm_ref, os.path.join(a.out, "vllm_commits.tsv.gz"), True)
    dump(a.sglang, a.sglang_ref, os.path.join(a.out, "sglang_commits.tsv.gz"), not a.sglang_no_numstat)
    notes(a.notes, os.path.join(a.out, "release_notes.json"))


if __name__ == "__main__":
    main()
