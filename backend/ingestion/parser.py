"""Text cleaning and section detection for ingested documents (spec section 5)."""
from __future__ import annotations

import re

# Matches the "LABEL: text" pattern produced by structured PubMed abstracts
# (see ingestion/loader.py._parse_pubmed_xml), e.g. "BACKGROUND: ...",
# "METHODS: ...", "RESULTS: ...", "CONCLUSIONS: ...". Anchored to the start
# of the string or right after whitespace (not just a line start) so labels
# packed onto the same line/paragraph are still detected correctly.
_SECTION_LABEL_RE = re.compile(r"(?:^|(?<=\s))([A-Z][A-Z \-/]{2,40}):\s")

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip stray control characters without
    altering the scientific content of the passage."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def detect_sections(text: str) -> list[tuple[str, str]]:
    """Split a (possibly structured) abstract into (section_name, section_text)
    pairs. Falls back to a single ('abstract', text) section when no
    structured labels are present - avoids fabricating structure that isn't
    in the source document.
    """
    flat_text = " ".join(l.strip() for l in text.split("\n") if l.strip())
    matches = list(_SECTION_LABEL_RE.finditer(flat_text))

    if not matches:
        return [("abstract", flat_text)] if flat_text else []

    sections: list[tuple[str, str]] = []
    # Anything before the first label (rare, but keep it rather than drop it).
    if matches[0].start() > 0:
        preamble = flat_text[: matches[0].start()].strip()
        if preamble:
            sections.append(("abstract", preamble))

    for i, match in enumerate(matches):
        label = match.group(1).strip().lower()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(flat_text)
        body = flat_text[body_start:body_end].strip()
        if body:
            sections.append((label, body))

    return sections
