"""month_review.py 的回归测试：python3 -m unittest test_month_review -v

数字全部钉在 data/ 快照上（vllm 0d6f92a、sglang 1ab008f5，同一个 30 天窗口）。
快照一换，这里的计数就要跟着改，这是故意的：正文引用的每个数都从这里来。
"""
import contextlib
import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import month_review as mr  # noqa: E402

V = mr.load("vllm_commits.tsv.gz")
S = mr.load("sglang_commits.tsv.gz")


def run(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a)
    return buf.getvalue()


class Snapshot(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(V), 1942)
        self.assertEqual(len(S), 2135)
        self.assertEqual(len({r["author"] for r in V}), 658)
        self.assertEqual(len({r["author"] for r in S}), 413)

    def test_every_vllm_commit_is_a_squashed_pr(self):
        self.assertTrue(all(r["pr"] for r in V))
        self.assertEqual(len({r["pr"] for r in V}), 1942)

    def test_inside_window(self):
        lo, hi = (mr.dt.datetime.fromisoformat(x.replace("Z", "+00:00")) for x in mr.WINDOW)
        for rows in (V, S):
            self.assertTrue(all(lo <= r["dt"] <= hi for r in rows))

    def test_sglang_has_no_numstat(self):
        # blob:none 克隆，numstat 要回源，导出时没取；scale 不能打印 0 行增删
        self.assertEqual(sum(r["ins"] + r["del"] for r in S), 0)
        out = run(mr.scale, V, S)
        self.assertEqual(out.count("增删行"), 1)


class Parsing(unittest.TestCase):
    def test_tags_only_leading(self):
        self.assertEqual(mr.tags("[Perf][DSv4.1] DeepSeek decode [WIP] (#1)"), ["Perf", "DSv4.1"])
        self.assertEqual(mr.tags(" [Fix] [AMD] moe: x"), ["Fix", "AMD"])
        self.assertEqual(mr.tags('Revert "[Docs] x" (#2)'), [])

    def test_pr_number_is_trailing(self):
        r = next(r for r in V if r["title"].startswith('Revert "[Docs] Add train-inference'))
        self.assertEqual(r["pr"], 62186)          # 不是被 revert 的 61551

    def test_ai_regex(self):
        rx = re.compile(mr.AI_NAMES, re.I)
        self.assertTrue(rx.search("Co-authored-by: Claude Opus 5.5 <noreply@anthropic.com>"))
        self.assertTrue(rx.search("co-authored-by:vllm-agent <a@b>"))
        self.assertFalse(rx.search("Co-authored-by: Alice <alice@x>"))
        self.assertFalse(rx.search("Signed-off-by: Claude Shannon <c@x>"))


class Scenarios(unittest.TestCase):
    def test_ai_share(self):
        self.assertEqual(sum(r["ai"] for r in V), 1109)
        self.assertEqual(sum(r["ai"] for r in S), 457)

    def test_ai_ramps_in_the_second_half(self):
        out = run(mr.ai, V, S)
        w = [int(x) for x in re.findall(r"W\d (\d+)%", out.split("-- SGLang")[0])]
        self.assertEqual(len(w), 5)
        self.assertLess(w[0], 50)
        self.assertGreater(w[3], 70)

    def test_weekend_local_time(self):
        self.assertEqual(sum(1 for r in V if r["dt"].weekday() >= 5), 242)
        self.assertEqual(sum(1 for r in S if r["dt"].weekday() >= 5), 444)

    def test_themes_overlap(self):
        hits = {k: {r["pr"] for r in h} for k, _, h in mr.theme_hits(V)}
        self.assertEqual(len(hits["mrv2"]), 52)
        self.assertEqual(len(hits["spec"]), 155)
        self.assertIn(61868, hits["ds"])
        # #61482 同时是 DeepSeek 和 ROCm：桶不能相加
        self.assertIn(61482, hits["ds"] & hits["rocm"])
        self.assertGreater(sum(map(len, hits.values())), len(set().union(*hits.values())))

    def test_models_dsv4_excludes_41(self):
        rx41 = re.compile(dict(mr.MODELS)["dsv4.1"], re.I)
        rx4 = re.compile(dict(mr.MODELS)["dsv4"], re.I)
        for t in ("[DSv4.1] x", "DeepSeek-V4.1-Flash", "ds-v4.1 fix"):
            self.assertTrue(rx41.search(t))
            self.assertFalse(rx4.search(t))
        self.assertTrue(rx4.search("DeepSeek-V4-Pro on MI355X"))

    def test_churn_top(self):
        out = run(mr.churn, V, S)
        self.assertRegex(out, r"vllm/models\s+172591")
        self.assertIn("#61302 +44859/-6", out)
        self.assertIn("#60517 +32/-4797", out)

    def test_coverage_split(self):
        out = run(mr.coverage, V, S, "2026-10-12")
        self.assertIn("合计覆盖 333 个（17.1%）", out)
        a = [int(x) for x in re.findall(r"：\s*(\d+) 个，覆盖 (\d+)", out)[0]]
        b = [int(x) for x in re.findall(r"：\s*(\d+) 个，覆盖 (\d+)", out)[1]]
        self.assertEqual(a[0] + b[0], 1942)
        self.assertEqual((a[1], b[1]), (323, 10))

    def test_coverage_moves_with_split(self):
        early = run(mr.coverage, V, S, "2026-10-05")
        late = run(mr.coverage, V, S, "2026-10-18")
        pct = lambda s: float(re.findall(r"以后：\s*\d+ 个，覆盖 \d+（([\d.]+)%", s)[0])
        self.assertGreater(pct(early), pct(late))

    def test_grep(self):
        out = run(mr.grep, V, "lookahead block hash|skip drafting")
        self.assertIn("#61589", out)
        self.assertIn("#62784", out)

    def test_main_all_runs(self):
        out = run(mr.main, ["--scenario", "all"])
        for sc in mr.SCEN:
            self.assertIn(f"== {sc}", out)


if __name__ == "__main__":
    unittest.main()
