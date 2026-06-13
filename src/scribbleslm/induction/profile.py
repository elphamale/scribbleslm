"""Profile data model + segmentation executor.

A Profile is an ordered list of hierarchy Levels (coarse -> fine), each a regex
matching a heading line (group 1 = label). Executing a profile walks the text,
tracks the active label at each level, and emits Segments between heading
boundaries, each carrying a breadcrumb of the active hierarchy.

Induced/synthesized regexes run through the `regex` module with a hard per-pattern
timeout — a hallucinated catastrophic-backtracking pattern is a self-DoS, not a
theoretical risk.
"""
from __future__ import annotations

import re as _re
from dataclasses import dataclass, field

try:
    import regex as _regex
except ImportError:  # pragma: no cover
    _regex = None

REGEX_TIMEOUT = 0.2  # seconds per pattern match


@dataclass
class Level:
    name: str       # e.g. "стаття"
    pattern: str    # regex matched against a stripped line; group(1) = label


@dataclass
class Profile:
    name: str
    levels: list[Level]
    fingerprint: str = ""
    rung: str = ""


@dataclass
class Segment:
    start: int          # char offset (inclusive)
    end: int            # char offset (exclusive)
    breadcrumb: str     # "розділ I › стаття 11"
    level_name: str     # finest level that opened this segment


def _match(pattern: str, line: str):
    """Anchored match with a per-pattern timeout (regex module if available)."""
    if _regex is not None:
        try:
            return _regex.match(pattern, line, timeout=REGEX_TIMEOUT)
        except (_regex.error, TimeoutError):
            return None
    try:
        return _re.match(pattern, line)
    except _re.error:
        return None


def find_headings(text: str, profile: Profile) -> list[tuple[int, int, str]]:
    """Return [(char_offset, level_index, label)] for every heading line."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            for li, lvl in enumerate(profile.levels):
                m = _match(lvl.pattern, stripped)
                if m:
                    label = (m.group(1) if m.groups() else stripped).strip()
                    out.append((pos, li, f"{lvl.name} {label}"))
                    break
        pos += len(line)
    return out


def segment(text: str, profile: Profile) -> list[Segment]:
    headings = find_headings(text, profile)
    if not headings:
        return []
    segs: list[Segment] = []
    active: dict[int, str] = {}
    for k, (off, lvl_i, label) in enumerate(headings):
        active[lvl_i] = label
        for finer in [l for l in active if l > lvl_i]:
            del active[finer]
        end = headings[k + 1][0] if k + 1 < len(headings) else len(text)
        crumb = " › ".join(active[l] for l in sorted(active))
        segs.append(Segment(off, end, crumb, profile.levels[lvl_i].name))
    return segs


# --- YAML (de)serialization for the profile cache -------------------------

def to_dict(p: Profile) -> dict:
    return {"name": p.name, "rung": p.rung, "fingerprint": p.fingerprint,
            "levels": [{"name": l.name, "pattern": l.pattern} for l in p.levels]}


def from_dict(d: dict) -> Profile:
    return Profile(name=d["name"], rung=d.get("rung", ""), fingerprint=d.get("fingerprint", ""),
                   levels=[Level(l["name"], l["pattern"]) for l in d["levels"]])
