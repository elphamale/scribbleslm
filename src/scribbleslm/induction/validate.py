"""Validation gate.

A profile must produce a plausible segmentation before we trust it:
- enough segments (not 1-2)
- >=95% character coverage from the first heading onward
- not over-fragmented (didn't match body lines as headings)
- NO oversized segment: a single segment larger than SEG_CHAR_CEILING (~33K tokens)
  is implausible for any one section and FAILS the gate. This is the signal that
  drives regional fallback (a region the profile couldn't segment — e.g. ПКУ's
  transitional tail that abandons Стаття for підрозділ/numbered points).

`oversized` lists the offending segment indices so the ladder can re-induce them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .profile import Profile, Segment, segment

# ~33K tokens at ~1.8 Cyrillic chars/token. A larger single segment means the
# profile failed to find structure in that region, not that the section is huge.
SEG_CHAR_CEILING = 60_000


@dataclass
class GateResult:
    passed: bool
    score: dict
    segments: list[Segment]
    oversized: list[int]   # indices of segments exceeding the ceiling


def validate(text: str, profile: Profile, *, min_coverage: float = 0.95,
             min_segments: int = 3, seg_char_ceiling: int = SEG_CHAR_CEILING) -> GateResult:
    segs = segment(text, profile)
    n = len(segs)
    if n < min_segments:
        return GateResult(False, {"reason": "too few segments", "segments": n}, segs, [])

    sizes = [s.end - s.start for s in segs]
    median = sorted(sizes)[n // 2]
    oversized = [i for i, c in enumerate(sizes) if c > seg_char_ceiling]
    first = segs[0].start
    coverage = (len(text) - first) / max(1, len(text))
    too_fragmented = n > max(4, len(text) // 200)

    passed = coverage >= min_coverage and not too_fragmented and not oversized
    score = {"segments": n, "coverage": round(coverage, 3), "median_chars": median,
             "max_chars": max(sizes), "oversized": len(oversized),
             "too_fragmented": too_fragmented}
    return GateResult(passed, score, segs, oversized)


def otherwise_sound(text: str, gate: GateResult) -> bool:
    """True when a failing gate's only real problem is oversized region(s) — i.e.
    coverage is fine and it isn't over-fragmented. Such a profile is worth salvaging
    via regional fallback rather than discarding."""
    return (bool(gate.oversized)
            and gate.score.get("coverage", 0) >= 0.95
            and not gate.score.get("too_fragmented", False))
