"""Induction ladder + profile-refinement rule.

Ladder: rung 2 (cache) -> rung 1 (format-native) -> rung 3 (line mining) ->
rung 4 (LLM synthesis) -> rung 5 (semantic) -> rung 6 (token fallback).

Profile-refinement rule (designed against the 5-code table):
  (a) cache hits run through the FULL validation gate — never trusted blindly;
  (b) gate failure due to OVERSIZED regions (otherwise sound) demotes the hit and
      triggers REGIONAL re-induction of those regions;
  (c) the result is cached as a depth-VARIANT keyed by fingerprint + levels-found,
      so e.g. ПКУ caches its стаття+розділ+підрозділ variant and stops rediscovering
      підрозділ on every ingest;
  (d) alignment-% is NEVER a control signal here (Tax legitimately sits at 75% and
      re-inducing cannot raise it) — it is reported by the caller for observability.

Regional fallback is best-effort: a genuinely flat region (digit-led numbered points)
that won't sub-segment is kept and token-split downstream at ingest.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cache import ProfileCache, fingerprint, variant_key
from .linemine import count_signatures, mine
from .profile import Level, Profile, Segment
from .validate import GateResult, SEG_CHAR_CEILING, otherwise_sound, validate

_MAX_REGION_DEPTH = 3


@dataclass
class Induction:
    profile: Profile | None        # None => token fallback (rung 6)
    segments: list[Segment]
    rung: str
    gate: dict


def _regional(text: str, gate: GateResult, depth: int = 0) -> tuple[list[Segment], dict[str, Level]]:
    """Re-induce each oversized segment independently and splice. Returns the new
    segments and the set of structural Levels discovered while doing so (used to
    build the refined depth-variant)."""
    out: list[Segment] = []
    discovered: dict[str, Level] = {}
    for i, s in enumerate(gate.segments):
        if i in gate.oversized and depth < _MAX_REGION_DEPTH:
            span = text[s.start:s.end]
            sub = mine(span, min_count=3)
            if sub:
                g = validate(span, sub)
                if len(g.segments) > 1:
                    for lvl in sub.levels:
                        discovered[lvl.name] = lvl
                    if g.oversized:
                        subsegs, subdisc = _regional(span, g, depth + 1)
                        discovered.update(subdisc)
                    else:
                        subsegs = g.segments
                    for ss in subsegs:
                        crumb = f"{s.breadcrumb} › {ss.breadcrumb}" if s.breadcrumb else ss.breadcrumb
                        out.append(Segment(s.start + ss.start, s.start + ss.end, crumb, ss.level_name))
                    continue
        out.append(s)
    return out, discovered


def _refined_profile(text: str, base: Profile, discovered: dict[str, Level]) -> Profile:
    """Merge base levels with regionally-discovered levels, ordered coarse->fine by
    document frequency (rarer keyword = higher level)."""
    levels = {l.name: l for l in base.levels}
    levels.update(discovered)
    counts, _ = count_signatures(text)
    freq = {name: sum(v for (kw, _cls), v in counts.items() if kw == name) for name in levels}
    ordered = sorted(levels.values(), key=lambda l: freq.get(l.name, 0))
    return Profile(name="refined", levels=ordered,
                   fingerprint=fingerprint(text), rung="refined")


def _rescore(text: str, segs: list[Segment]) -> dict:
    sizes = [s.end - s.start for s in segs] or [0]
    return {"segments": len(segs), "max_chars": max(sizes),
            "oversized": sum(1 for c in sizes if c > SEG_CHAR_CEILING)}


def _accept(text: str, profile: Profile, rung: str, cache: ProfileCache | None) -> Induction | None:
    """Run the gate; on a clean pass, accept. On an oversized-but-sound failure, do
    regional fallback, cache the refined depth-variant, and accept. Else reject."""
    g = validate(text, profile)
    if g.passed:
        return Induction(profile, g.segments, rung, g.score)
    if otherwise_sound(text, g):
        segs, discovered = _regional(text, g)
        refined = _refined_profile(text, profile, discovered)
        if cache is not None:
            cache.put(refined.fingerprint, refined)
        return Induction(refined, segs, rung + "-regional", _rescore(text, segs))
    return None


def induce(text: str, *, cache: ProfileCache | None = None,
           format_native=None, llm_synth=None, semantic=None) -> Induction:
    fp = fingerprint(text)
    cands = cache.candidates(fp) if cache is not None else []

    # rung 2, pass 1 — SIMPLEST cached variant that passes the gate cleanly.
    # (A doc uses the minimal profile that fits it: ККУ -> стаття+розділ, not the
    # richest variant some other statute cached.)
    for cand in sorted(cands, key=lambda p: len(p.levels)):
        g = validate(text, cand)
        if g.passed:
            return Induction(cand, g.segments, "2-cache", g.score)

    # rung 2, pass 2 — no clean fit: RICHEST salvageable variant via regional
    # fallback (ПКУ -> its підрозділ depth-variant, without re-mining structure).
    for cand in cands:  # candidates() is richest-first
        g = validate(text, cand)
        if otherwise_sound(text, g):
            segs, discovered = _regional(text, g)
            refined = _refined_profile(text, cand, discovered)
            if cache is not None:
                cache.put(refined.fingerprint, refined)
            return Induction(refined, segs, "2-cache-regional", _rescore(text, segs))

    # rung 1 — format-native
    if format_native is not None:
        p = format_native(text)
        if p and (r := _accept(text, p, "1-format", cache)):
            if cache:
                cache.put(fp, p)
            return r

    # rung 3 — line-shape mining
    p = mine(text)
    if p and (r := _accept(text, p, "3-linemine", cache)):
        if cache:
            cache.put(fp, p)
        return r

    # rung 4 — LLM synthesis
    if llm_synth is not None:
        p = llm_synth(text)
        if p and (r := _accept(text, p, "4-llm", cache)):
            if cache:
                cache.put(fp, p)
            return r

    # rung 5 — semantic segmentation
    if semantic is not None:
        segs = semantic(text)
        if segs:
            return Induction(None, segs, "5-semantic", {"segments": len(segs)})

    # rung 6 — token fallback
    return Induction(None, [], "6-token", {})
