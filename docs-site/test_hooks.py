"""Unit tests for docs-site/hooks.py: python3 -m unittest discover -s docs-site"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import hooks


class FakeFiles:
    def __init__(self, paths):
        self.paths = set(paths)

    def get_file_from_path(self, path):
        return path if path in self.paths else None


def page(src):
    return SimpleNamespace(file=SimpleNamespace(src_uri=src))


class HooksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "book").mkdir()
        (root / "02-ledger").mkdir()
        (root / "02-ledger" / "sim.py").write_text("")
        (root / "mkdocs.yml").write_text("")
        self.config = {
            "config_file_path": str(root / "mkdocs.yml"),
            "repo_url": "https://github.com/o/r",
            "extra": {"repo_prefix": "book"},
        }
        self.files = FakeFiles({"02-ledger.md", "assets/a.svg"})

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, md, src="01-x.md"):
        return hooks.on_page_markdown(md, page(src), self.config, self.files)

    def test_label(self):
        self.assertEqual(hooks._label("VLLM-02｜上手｜推理账本"), ("02 推理账本", "上手"))
        self.assertEqual(
            hooks._label("运维篇 01｜版本、安装与实验环境：先用 nginx -V 认清手里的二进制"),
            ("01 版本、安装与实验环境", None),
        )
        self.assertEqual(hooks._label("Zero To SGLang"), ("Zero To SGLang", None))

    def test_inline_math(self):
        self.assertEqual(self.render("除以 $`\\sqrt{d}`$ 即可"), "除以 \\(\\sqrt{d}\\) 即可")

    def test_dollar_code_spans_are_not_math(self):
        md = "正则 `^a$` 和 `$b` 都是代码"
        self.assertEqual(self.render(md), md)

    def test_math_untouched_in_fence(self):
        md = "```text\n$`x`$\n```"
        self.assertEqual(self.render(md), md)

    def test_blank_line_before_list_and_table(self):
        self.assertEqual(self.render("清单\n- a\n- b"), "清单\n\n- a\n- b")
        self.assertEqual(self.render("表\n| a |"), "表\n\n| a |")
        self.assertEqual(self.render("> **清单**\n> - a\n> - b"), "> **清单**\n>\n> - a\n> - b")

    def test_no_blank_line_inside_list(self):
        md = "- a\n- b\n  continued\n1. c"
        self.assertEqual(self.render(md), md)

    def test_links(self):
        self.assertEqual(self.render("[x](02-ledger.md#a)"), "[x](02-ledger.md#a)")
        self.assertEqual(self.render("![f](assets/a.svg)"), "![f](assets/a.svg)")
        self.assertEqual(
            self.render("[d](../02-ledger)"), "[d](https://github.com/o/r/tree/main/02-ledger)"
        )
        self.assertEqual(
            self.render("[s](../02-ledger/sim.py#L3)"),
            "[s](https://github.com/o/r/blob/main/02-ledger/sim.py#L3)",
        )
        self.assertEqual(
            self.render("由 [`02-ledger/sim.py`](../02-ledger/sim.py) 链接"),
            "由 [`02-ledger/sim.py`](https://github.com/o/r/blob/main/02-ledger/sim.py) 链接",
        )
        self.assertEqual(
            self.render('<a href="../02-ledger/sim.py">sim</a> <img src="assets/a.svg">'),
            '<a href="https://github.com/o/r/blob/main/02-ledger/sim.py">sim</a> <img src="assets/a.svg">',
        )
        self.assertEqual(self.render("[m](../missing.py)"), "[m](../missing.py)")
        self.assertEqual(self.render("`[c](../02-ledger)`"), "`[c](../02-ledger)`")
        self.assertEqual(self.render("[u](https://x.org/a)"), "[u](https://x.org/a)")


if __name__ == "__main__":
    unittest.main()
