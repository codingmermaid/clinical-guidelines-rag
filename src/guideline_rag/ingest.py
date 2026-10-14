"""Download NICE guidelines (politely, once) and cache the raw HTML locally.

Parsing lives further down in this file (see ``parse_guideline``). Keeping the
download step separate from parsing means we hit NICE's servers only once per
page; every later parser change just re-reads the cached files.
"""

from __future__ import annotations

import re
import time
import urllib.robotparser
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

from guideline_rag.config import load_config, project_path

# Pages whose side menu lists all chapters. Not every guideline has the first one
# (NG136 does not), so we try them in order and keep the first that exists.
INDEX_CHAPTERS = ["Using-this-guideline", "Recommendations"]


def load_sources() -> dict:
    """Read data/sources.yaml (the list of guidelines to download)."""
    with open(project_path(load_config()["paths"]["sources"]), encoding="utf-8") as f:
        return yaml.safe_load(f)


def chapter_url(base_url: str, code: str, chapter: str) -> str:
    return f"{base_url}/{code.lower()}/chapter/{chapter}"


def raw_path(code: str, chapter: str) -> Path:
    return project_path(load_config()["paths"]["raw"]) / f"{code}__{chapter}.html"


def list_chapters(index_html: str, code: str, exclude: list[str]) -> list[str]:
    """Read the guideline's chapter menu and keep only chapters that hold recommendations.

    Why: NICE does not use one layout for every guideline, so we follow the menu
    instead of hard-coding chapter names per guideline.
    """
    soup = BeautifulSoup(index_html, "lxml")
    prefix = f"/guidance/{code.lower()}/chapter/"
    chapters: list[str] = []
    for a in soup.select("nav.stacked-nav a"):
        href = a.get("href", "")
        if href.startswith(prefix):
            name = href[len(prefix):].split("#")[0]
            if name not in exclude and name not in chapters:
                chapters.append(name)
    return chapters


class PoliteFetcher:
    """Downloads pages with a User-Agent, robots.txt check, delay and local cache."""

    def __init__(self, base_url: str):
        cfg = load_config()["ingest"]
        self.delay, self.ua = cfg["delay_seconds"], cfg["user_agent"]
        root = "/".join(base_url.split("/")[:3])
        resp = requests.get(f"{root}/robots.txt", headers={"User-Agent": self.ua}, timeout=30)
        resp.raise_for_status()
        self.robots = urllib.robotparser.RobotFileParser()
        self.robots.parse(resp.text.splitlines())
        self._last_request = 0.0

    def get(self, url: str, cache: Path, force: bool = False) -> str:
        if cache.exists() and not force:
            return cache.read_text(encoding="utf-8")
        if not self.robots.can_fetch(self.ua, url):
            raise PermissionError(f"robots.txt disallows {url}")
        wait = self.delay - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)  # be polite: at least 2 s between requests
        resp = requests.get(url, headers={"User-Agent": self.ua}, timeout=60)
        self._last_request = time.time()
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        print(f"download {cache.name} ({len(resp.text) // 1024} KB)")
        return resp.text


def download_all(force: bool = False) -> dict[str, list[str]]:
    """Download every chapter of every configured guideline; return {code: [chapters]}."""
    sources = load_sources()
    base = sources["base_url"]
    fetcher = PoliteFetcher(base)
    result: dict[str, list[str]] = {}
    for g in sources["guidelines"]:
        code = g["code"]
        index = None
        for candidate in INDEX_CHAPTERS:
            try:
                index = fetcher.get(chapter_url(base, code, candidate), raw_path(code, candidate), force)
                break
            except requests.HTTPError as err:
                if err.response is None or err.response.status_code != 404:
                    raise  # only "page does not exist" means "try the next candidate"
        if index is None:
            raise RuntimeError(f"Could not read the chapter menu for {code}")
        chapters = list_chapters(index, code, sources["exclude_chapters"])
        for chapter in chapters:
            fetcher.get(chapter_url(base, code, chapter), raw_path(code, chapter), force)
        result[code] = chapters
        print(f"{code}: {len(chapters)} chapter(s): {', '.join(chapters)}")
    return result


# ---------------------------------------------------------------------------
# Parsing: turn the cached HTML into clean records with metadata.
# ---------------------------------------------------------------------------


class Record(BaseModel):
    """One piece of guideline text plus the metadata needed to cite it."""

    record_id: str  # e.g. NG106-1.4.3, NG115-table-2, NG28-ctx-14
    guideline_code: str
    guideline_title: str
    organization: str = "NICE"
    section_path: str  # e.g. "Initial medicines > 1.13 People with ... > Treatment"
    recommendation_id: str | None = None  # only for numbered recommendations
    last_updated: str  # ISO date taken from the page
    source_url: str  # links to the exact section (page anchor)
    content_type: str  # recommendation | table | context
    text: str


# Page furniture that is not guideline content.
_SKIP_CLASSES = {"prev-next", "back-to-top", "hide-print", "in-page-nav", "stacked-nav"}
_HEADINGS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _clean(text: str) -> str:
    """Collapse runs of whitespace (HTML source has many line breaks) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(el: Tag) -> str:
    """Convert a block of HTML to plain text, keeping paragraphs and bullet lists readable."""
    lines: list[str] = []

    def block(node: Tag, indent: int = 0) -> None:
        inline_buf: list[str] = []

        def flush() -> None:
            t = _clean("".join(inline_buf))
            if t:
                lines.append("  " * indent + t)
            inline_buf.clear()

        for ch in node.children:
            if not isinstance(ch, Tag):
                inline_buf.append(str(ch))
            elif ch.name in ("ul", "ol"):
                flush()
                for n, li in enumerate(ch.find_all("li", recursive=False), 1):
                    marker = "- " if ch.name == "ul" else f"{n}. "
                    sub = html_to_text(li).split("\n")
                    lines.append("  " * indent + marker + sub[0])
                    lines.extend("  " * (indent + 1) + s for s in sub[1:])
            elif ch.name in ("p", "div", "details", "summary", "h2", "h3", "h4", "h5", "figure", "li"):
                flush()
                block(ch, indent)
            elif ch.name == "br":
                flush()
            elif ch.name == "table":
                continue  # tables are extracted as separate records
            else:
                inline_buf.append(ch.get_text(""))
        flush()

    block(el)
    return "\n".join(lines)


def table_to_markdown(table: Tag) -> tuple[str, str]:
    """Return (caption, markdown) for an HTML table. The header row is always kept."""
    cap = table.find("caption")
    caption = _clean(cap.get_text(" ")) if cap else ""

    def cells(row: Tag) -> list[str]:
        out = []
        for c in row.find_all(["th", "td"]):
            text = "; ".join(html_to_text(c).split("\n")).replace("|", "\\|")
            out.append(_clean(text))
        return out

    head_rows = [cells(r) for r in table.select("thead tr")]
    body_rows = [cells(r) for r in table.find_all("tr") if not r.find_parent("thead")]
    if not head_rows and body_rows:  # no <thead>: treat the first row as the header
        head_rows, body_rows = [body_rows[0]], body_rows[1:]
    header = head_rows[-1] if head_rows else []
    width = max([len(header)] + [len(r) for r in body_rows])
    pad = lambda r: r + [""] * (width - len(r))  # noqa: E731
    md = ["| " + " | ".join(pad(header)) + " |", "|" + " --- |" * width]
    md += ["| " + " | ".join(pad(r)) + " |" for r in body_rows]
    return caption, "\n".join(md)


_NUMBER = re.compile(r"^\d+(\.\d+)+$")  # e.g. 1.4.3


def _recommendation_number(article: Tag) -> str | None:
    """Return '1.4.3' if this <article> is a numbered recommendation, else None.

    NICE marks the number with class ``recommendation__number`` on some
    guidelines and with a bare <h4> on others, so we look at the heading text.
    """
    # The heading level grows with nesting depth (h4 ... h6), so accept any level.
    heading = article.find(["h2", "h3", "h4", "h5", "h6"], recursive=False)
    text = _clean(heading.get_text()) if heading else ""
    return text if _NUMBER.match(text) else None


def _page_meta(soup: BeautifulSoup) -> tuple[str, str]:
    """Read the guideline title and the 'Last updated' date from the page header."""
    h1 = soup.select_one("h1.page-header__heading")
    title = _clean(h1.get_text(" ")) if h1 else ""
    times = soup.select(".page-header time[datetime]")
    return title, (times[-1]["datetime"] if times else "")  # the last <time> is "Last updated"


def parse_chapter(html: str, code: str, chapter: str, base_url: str, counters: dict) -> list[Record]:
    """Parse one cached chapter page into recommendation, table and context records."""
    soup = BeautifulSoup(html, "lxml")
    title, updated = _page_meta(soup)
    root = soup.select_one("div.chapter") or soup.find("main")
    page_url = chapter_url(base_url, code, chapter)
    records: list[Record] = []
    path: dict[int, str] = {}  # heading level -> heading text, i.e. where we are in the document
    buffer: list[str] = []  # loose paragraphs waiting to become one context record
    anchor = {"id": ""}  # id of the current section, used for deep links

    def section_path() -> str:
        parts = [path[k] for k in sorted(path) if path[k] != "Recommendations"]
        return " > ".join(parts) or chapter.replace("-", " ")  # intro text may sit above any heading

    def add(kind: str, rid: str, text: str, rec_no: str | None, frag: str) -> None:
        if not text.strip():
            return
        records.append(Record(
            record_id=rid, guideline_code=code, guideline_title=title, section_path=section_path(),
            recommendation_id=rec_no, last_updated=updated, content_type=kind, text=text.strip(),
            source_url=f"{page_url}#{frag}" if frag else page_url))

    def flush_context() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            counters["ctx"] = counters.get("ctx", 0) + 1
            add("context", f"{code}-ctx-{counters['ctx']}", text, None, anchor["id"])

    def add_table(table: Tag, frag: str) -> None:
        flush_context()
        caption, md = table_to_markdown(table)
        counters["table"] = counters.get("table", 0) + 1
        add("table", f"{code}-table-{counters['table']}", (caption + "\n\n" if caption else "") + md,
            None, table.get("id") or frag)

    def walk(node: Tag) -> None:
        for ch in node.children:
            if not isinstance(ch, Tag):
                continue
            classes = set(ch.get("class", []))
            if classes & _SKIP_CLASSES or ch.name in ("nav", "script", "style"):
                continue
            if ch.name in _HEADINGS:
                flush_context()
                level = _HEADINGS[ch.name]
                path[level] = _clean(ch.get_text(" "))
                for deeper in [k for k in path if k > level]:
                    del path[deeper]
            elif ch.name == "article" and _recommendation_number(ch):
                flush_context()
                number = _recommendation_number(ch)
                # NG106/NG115/NG136 wrap the text in .recommendation__body; NG28 uses a plain <div>.
                body = ch.select_one(".recommendation__body") or ch.find("div", recursive=False) or ch
                add("recommendation", f"{code}-{number}" if number else f"{code}-rec-{ch.get('id')}",
                    html_to_text(body), number, ch.get("id", ""))
                for t in body.find_all("table"):  # a table inside a recommendation stays separate
                    add_table(t, ch.get("id", ""))
            elif ch.name == "table":
                add_table(ch, anchor["id"])
            elif ch.name == "div" and "informaltable" in classes:
                for t in ch.find_all("table"):
                    add_table(t, anchor["id"])
            elif ch.name == "div" and "section" in classes:
                flush_context()
                saved_path, saved_anchor = dict(path), anchor["id"]  # headings must not leak out of a section
                anchor["id"] = ch.get("id") or anchor["id"]
                walk(ch)
                flush_context()
                path.clear()
                path.update(saved_path)
                anchor["id"] = saved_anchor
            elif ch.name == "div" and ch.find(["article", "table"]):
                # A wrapper (e.g. a collapsible "reveal" panel) around real recommendations:
                # look inside it instead of flattening it into plain text.
                walk(ch)
            elif ch.name == "div" and "panel" in classes:
                flush_context()
                buffer.append(html_to_text(ch))
                flush_context()
            else:
                text = html_to_text(ch)
                if text:
                    buffer.append(text)

    walk(root)
    flush_context()
    return records


def parse_guideline(code: str, chapters: list[str], base_url: str) -> list[Record]:
    """Parse all cached chapters of one guideline; counters keep table/context ids unique."""
    counters: dict = {}
    out: list[Record] = []
    for chapter in chapters:
        html = raw_path(code, chapter).read_text(encoding="utf-8")
        out += parse_chapter(html, code, chapter, base_url, counters)
    return out


def parse_all() -> list[Record]:
    """Download (cached) and parse every guideline, write records.jsonl, print a summary."""
    sources = load_sources()
    chapters_by_code = download_all()
    records: list[Record] = []
    for g in sources["guidelines"]:
        records += parse_guideline(g["code"], chapters_by_code[g["code"]], sources["base_url"])
    out = project_path(load_config()["paths"]["processed"]) / "records.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    print(f"\n{'guideline':10} {'recommendations':>16} {'tables':>7} {'context':>8}")
    for g in sources["guidelines"]:
        mine = [r for r in records if r.guideline_code == g["code"]]
        count = lambda k: sum(r.content_type == k for r in mine)  # noqa: E731
        print(f"{g['code']:10} {count('recommendation'):>16} {count('table'):>7} {count('context'):>8}")
    print(f"\nSaved {len(records)} records to {out}")
    return records


if __name__ == "__main__":
    parse_all()
