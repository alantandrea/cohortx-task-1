"""Parse PMC JATS .nxml files into structured text.

PMC NXML in this competition is JATS without XML namespaces. We pull the title,
abstract, keywords, and body sections (keyed by their cleaned titles), plus a
"methods/patient-selection" view that concatenates the sections most likely to
describe cohort selection. All extraction downstream reads from this structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from lxml import etree


def _local(el) -> str | None:
    """Local tag name, or None for comments/PIs."""
    t = el.tag
    if not isinstance(t, str):
        return None
    return t.split("}")[-1]


def _text(el) -> str:
    """Whitespace-normalised inner text of an element."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


# Section titles that typically carry cohort/patient-selection information.
_METHODS_RE = re.compile(
    r"method|material|patient|subject|particip|population|design|"
    r"cohort|selection|enroll|inclus|exclus|eligib|recruit|study sample",
    re.I,
)
_NUM_PREFIX_RE = re.compile(r"^\s*[\dIVX]+(\.\d+)*[\.\)]\s*", re.I)


def _clean_title(t: str) -> str:
    return _NUM_PREFIX_RE.sub("", (t or "").strip()).strip()


@dataclass
class Article:
    pmcid: str
    pmid: str = ""
    title: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)  # (clean_title, text)
    full_text: str = ""

    @property
    def methods_text(self) -> str:
        """Concatenated text of sections whose title looks methods/cohort-ish."""
        parts = [txt for title, txt in self.sections if _METHODS_RE.search(title)]
        return "\n".join(parts)

    @property
    def title_abstract(self) -> str:
        return f"{self.title}. {self.abstract}".strip()

    def section_by(self, pattern: str) -> str:
        rx = re.compile(pattern, re.I)
        return "\n".join(txt for title, txt in self.sections if rx.search(title))


def _iter_top_sections(body):
    """Yield (title, element) for top-level <sec> in body; fall back to whole body."""
    secs = [el for el in body if _local(el) == "sec"]
    if secs:
        for sec in secs:
            title_el = next((c for c in sec if _local(c) == "title"), None)
            yield _clean_title(_text(title_el)), sec
    else:
        yield "", body


def parse_file(path: str | Path) -> Article:
    path = Path(path)
    pmcid = re.sub(r"^PMC|\.nxml$", "", path.name)
    parser = etree.XMLParser(recover=True, huge_tree=True)
    try:
        root = etree.parse(str(path), parser).getroot()
    except Exception:
        return Article(pmcid=pmcid)
    if root is None:
        return Article(pmcid=pmcid)

    art = Article(pmcid=pmcid)

    title_el = next((e for e in root.iter() if _local(e) == "article-title"), None)
    art.title = _text(title_el)

    for aid in root.iter():
        if _local(aid) == "article-id" and (aid.get("pub-id-type") == "pmid"):
            txt = (aid.text or "").strip()
            if txt.isdigit():
                art.pmid = txt
                break

    # Abstract: prefer <abstract> in front matter (skip graphical/supplementary).
    for ab in root.iter():
        if _local(ab) == "abstract":
            atype = (ab.get("abstract-type") or "").lower()
            if atype in {"graphical", "teaser", "web-summary"}:
                continue
            txt = _text(ab)
            # Drop a leading "Abstract" heading that itertext pulls in.
            txt = re.sub(r"^\s*abstract\b[:\s]*", "", txt, flags=re.I).strip()
            art.abstract = txt
            if art.abstract:
                break

    art.keywords = [
        _text(k) for k in root.iter() if _local(k) == "kwd" and _text(k)
    ]

    body = next((e for e in root.iter() if _local(e) == "body"), None)
    if body is not None:
        for title, sec in _iter_top_sections(body):
            txt = _text(sec)
            # Drop the leading title token from the section body text.
            if title and txt.lower().startswith(title.lower()):
                txt = txt[len(title):].strip()
            if txt:
                art.sections.append((title, txt))

    # Full text view = title + abstract + body sections.
    art.full_text = "\n".join(
        [art.title, art.abstract] + [t for _, t in art.sections]
    ).strip()
    return art


@lru_cache(maxsize=4096)
def parse_cached(path: str) -> Article:
    return parse_file(path)
