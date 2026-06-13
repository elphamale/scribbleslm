"""Rung 3 — line-shape mining (Drain-style template induction, no LLM).

Reduce each heading-like line to a shape signature (leading keyword + a class for
the following token: <NUM> or <ROM>). Signatures that recur often at line starts
are boundary candidates; rarer recurring signatures sit at higher hierarchy levels.
Emit a Profile from the winning signature set.

Free, deterministic, no model. This is the rung that must carry Ukrainian statutes
on its own (acceptance test: cache empty + PROFILE_SYNTHESIS off).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .profile import Level, Profile

_ROMAN = re.compile(r"^[IVXLCDM]+$")
_NUM = re.compile(r"^\d+(?:[.\-]\d+)*$")
_KEYWORD = re.compile(r"^[^\W\d_][\w'-]*$", re.UNICODE)  # a word, no leading digit

# Ukrainian function words that precede numbers/dates in body text ("з 1 січня…",
# "до 31 грудня…") and must never be mistaken for structural heading keywords.
_STOPWORDS = {
    "з", "до", "від", "після", "перед", "протягом", "на", "у", "в", "за", "по",
    "під", "над", "при", "для", "понад", "близько", "не", "і", "та", "або", "що",
    "як", "коли", "якщо", "це", "усі", "всі", "цей", "цього", "станом",
}


def _second_class(tok: str) -> str | None:
    tok = tok.rstrip(".:)")
    if _NUM.match(tok):
        return "<NUM>"
    if _ROMAN.match(tok):
        return "<ROM>"
    return None


def count_signatures(text: str) -> tuple[Counter, dict]:
    """Heading-line shape signatures: (keyword.lower(), <NUM>|<ROM>) -> count,
    plus the original-case keyword for each signature."""
    counts: Counter = Counter()
    keyword_case: dict = {}
    for line in text.splitlines():
        s = line.strip()
        # headings are identified by their leading tokens; some carry an inline note
        # (e.g. "Розділ XX. TITLE {виключено …}"), so allow long lines rather than
        # skipping them — the keyword+number filter below is what actually selects.
        if not s or len(s) > 300:
            continue
        toks = s.split()
        if len(toks) < 2:
            continue
        kw = toks[0].rstrip(".")
        if not _KEYWORD.match(kw) or not kw[0].isupper():
            continue
        if kw.lower() in _STOPWORDS:   # a preposition before a date, not a heading
            continue
        cls = _second_class(toks[1])
        if cls is None:
            continue
        sig = (kw.lower(), cls)
        counts[sig] += 1
        keyword_case.setdefault(sig, kw)
    return counts, keyword_case


def mine(text: str, min_count: int = 5, max_levels: int = 4) -> Profile | None:
    counts, keyword_case = count_signatures(text)
    cands = [(sig, c) for sig, c in counts.items() if c >= min_count]
    if not cands:
        return None

    # most frequent signatures are the real structural headings; take the top set,
    # then order coarse -> fine by ascending frequency (rarer keyword = higher level).
    cands.sort(key=lambda x: -x[1])
    chosen = cands[:max_levels]
    chosen.sort(key=lambda x: x[1])

    levels: list[Level] = []
    for (kw_low, cls), _ in chosen:
        kw = keyword_case[(kw_low, cls)]
        numpat = r"(\d+(?:[.\-]\d+)*)" if cls == "<NUM>" else r"([IVXLCDM]+)"
        levels.append(Level(name=kw_low, pattern=rf"{re.escape(kw)}\s+{numpat}"))
    return Profile(name="line-mined", levels=levels, rung="3-linemine")
