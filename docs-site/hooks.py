"""MkDocs hooks for the online edition of the book.

The Markdown is written for GitHub, so a few things are adapted at build time;
the source files are never changed.

- nav: built from the chapter headings ("PREFIX-NN｜stage｜title" or "part NN｜title"),
  grouped by stage, or by sub-directory (extra.nav_sections) when chapters live in folders.
  extra.nav_stages ({stage: section name}) collects every chapter of a stage into one
  section, for series whose numbering interleaves stages.
- math: GitHub's inline $`...`$ becomes \\( ... \\) for pymdownx.arithmatex.
- blank lines: GitHub starts a list or table right after a paragraph line;
  Python-Markdown needs a blank line first, so one is inserted.
- links: relative Markdown and HTML links that leave the site (code folders, scripts,
  labs) point to the file on GitHub instead of a missing page.
- search: the lunr bundled in the search worker keys TokenSet states as label + child id
  with no delimiter, so a digit label can merge two different states; on a large index
  some queries then throw. A delimiter is added after the id in the built worker.
"""

import os
import re
from pathlib import Path
from urllib.parse import unquote

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(`+)(?:.+?)\1")
LINK = re.compile(r"(!?\[(?:[^\[\]]|\[[^\]]*\])*\]\()(<[^>]+>|[^)\s]+)((?:\s+\"[^\"]*\")?\))")
BLOCK_START = re.compile(r"^(?:[-*+]|\d{1,3}[.)])\s+\S|^\|")
NUMBER = re.compile(r"(\d+)\s*$")
HTML_LINK = re.compile(r"(\b(?:href|src)=\")([^\"]+)(\")")
QUOTE = re.compile(r"^((?:\s{0,3}>\s?)*)(.*)$")


def _heading(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _label(title):
    """Return (sidebar label, stage) for a chapter heading."""
    parts = [p.strip() for p in title.split("｜")]
    if len(parts) == 3:
        prefix, stage, text = parts
    elif len(parts) == 2:
        prefix, text = parts
        stage = None
    else:
        return title, None
    no = NUMBER.search(prefix)
    short = text.split("：")[0] if len(text) > 20 else text
    return (f"{no.group(1)} {short}" if no else short), stage


def _chapters(docs, folder=""):
    """Numbered chapters of one folder grouped by stage: [(stage, [{label: path}])]."""
    groups = []
    for md in sorted((docs / folder).glob("[0-9]*.md")):
        label, stage = _label(_heading(md))
        rel = f"{folder}/{md.name}" if folder else md.name
        if not groups or groups[-1][0] != stage:
            groups.append((stage, []))
        groups[-1][1].append({label: rel})
    return groups


def on_config(config):
    docs = Path(config["docs_dir"])
    index = next(n for n in ("index.md", "README.md") if (docs / n).exists())
    label, _ = _label(_heading(docs / index))
    nav = [{label if label[:1].isdigit() else "首页": index}]
    sections = config["extra"].get("nav_sections")
    if sections:
        for folder, name in sections.items():
            pages = [p for _, group in _chapters(docs, folder) for p in group]
            nav.append({name: pages})
    elif config["extra"].get("nav_stages"):
        by_stage = {}
        for stage, pages in _chapters(docs):
            by_stage.setdefault(stage, []).extend(pages)
        for stage, name in config["extra"]["nav_stages"].items():
            nav.append({name: by_stage.pop(stage)})
        if by_stage:
            raise ValueError(f"chapters with a stage missing from extra.nav_stages: {sorted(by_stage)}")
    else:
        for stage, pages in _chapters(docs):
            nav.append({stage: pages} if stage else pages[0])
    config["nav"] = nav
    return config


def _outbound(target, page, config, files):
    """GitHub URL for a relative link that does not resolve inside the site, else None."""
    if target.startswith("<"):
        target = target[1:-1]
    if re.match(r"^[a-z][a-z0-9+.-]*:|^#|^/", target, re.IGNORECASE):
        return None
    path, _, anchor = target.partition("#")
    rel = os.path.normpath(os.path.join(os.path.dirname(page.file.src_uri), unquote(path)))
    if not rel.startswith("..") and files.get_file_from_path(rel.replace(os.sep, "/")):
        return None
    root = Path(config["config_file_path"]).parent
    repo_path = os.path.normpath(os.path.join(config["extra"].get("repo_prefix", ""), rel))
    if repo_path.startswith("..") or not (root / repo_path).exists():
        return None
    kind = "tree" if (root / repo_path).is_dir() else "blob"
    branch = config["extra"].get("repo_branch", "main")
    url = f"{config['repo_url'].rstrip('/')}/{kind}/{branch}/{repo_path}"
    return url + (f"#{anchor}" if anchor else "")


def _inline(text, page, config, files):
    """Rewrite links that do not start inside a code span; a code span wrapped in $...$ is math."""
    for pattern in (LINK, HTML_LINK):
        code = [m.span() for m in CODE_SPAN.finditer(text)]

        def repl(m, code=code):
            if any(start <= m.start() < end for start, end in code):
                return m.group(0)
            url = _outbound(m.group(2), page, config, files)
            return f"{m.group(1)}{url}{m.group(3)}" if url else m.group(0)

        text = pattern.sub(repl, text)
    out, last = [], 0
    for span in CODE_SPAN.finditer(text):
        start, end = span.start(), span.end()
        if (
            len(span.group(1)) == 1
            and text[start - 1 : start] == "$"
            and text[end : end + 1] == "$"
        ):
            out.append(text[last : start - 1])
            out.append("\\(" + text[start + 1 : end - 1] + "\\)")
            last = end + 1
    out.append(text[last:])
    return "".join(out)


def on_page_markdown(markdown, page, config, files):
    out, fence, prev = [], None, ""
    for line in markdown.split("\n"):
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            out.append(line)
            prev = line
            continue
        if m:
            fence = m.group(1)
            out.append(line)
            prev = line
            continue
        line = _inline(line, page, config, files)
        quote, body = QUOTE.match(line).groups()
        prev_quote, prev_body = QUOTE.match(prev).groups()
        if (
            BLOCK_START.match(body)
            and quote.replace(" ", "") == prev_quote.replace(" ", "")
            and prev_body.strip()
            and not BLOCK_START.match(prev_body)
            and not prev_body.startswith((" ", "\t", "#"))
        ):
            out.append(quote.rstrip())
        out.append(line)
        prev = line
    return "\n".join(out)


LUNR_KEY = "e=e+s+o.id}return e"


def _patch_lunr(js):
    """Add a delimiter after the child id in lunr's TokenSet key; raise if the code moved."""
    if js.count(LUNR_KEY) != 1:
        raise ValueError("lunr TokenSet.toString not found in the search worker; recheck the patch")
    return js.replace(LUNR_KEY, 'e=e+s+o.id+"|"}return e')


def on_post_build(config):
    workers = list(Path(config["site_dir"], "assets", "javascripts", "workers").glob("search.*.js"))
    if not workers:
        raise ValueError("search worker not found")
    for path in workers:
        path.write_text(_patch_lunr(path.read_text(encoding="utf-8")), encoding="utf-8")
