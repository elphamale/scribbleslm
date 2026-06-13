"""Rung 2 — profile cache, keyed by STRUCTURAL FINGERPRINT (not content hash).

The fingerprint is the document's dominant line-shape signature(s), so a different
statute (ПКУ vs ККУ) hashes to the same key as the statute profile and reuses it,
while genuinely different document shapes miss and fall through the ladder.

Profiles persist as YAML. A shipped profiles/ dir provides pre-warmed, deletable
entries (e.g. ukrainian_statute.yaml); a user cache dir holds induced ones.
"""
from __future__ import annotations

from pathlib import Path

from .linemine import count_signatures
from .profile import Profile, from_dict, to_dict


def fingerprint(text: str, top: int = 1) -> str:
    """Dominant heading signature(s) as the cache key. top=1 keeps it tolerant so
    statutes share a key; raise `top` for stricter matching."""
    counts, _ = count_signatures(text)
    if not counts:
        return "none"
    return ";".join(f"{kw}|{cls}" for (kw, cls), _ in counts.most_common(top))


class ProfileCache:
    def __init__(self, cache_dir: str | Path, shipped_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shipped_dir = Path(shipped_dir) if shipped_dir else None
        self._mem: dict[str, Profile] = {}
        self._load_all()

    def _load_all(self) -> None:
        import yaml
        # shipped first, then user cache (user overrides shipped on key collision)
        for d in (self.shipped_dir, self.cache_dir):
            if not d or not d.exists():
                continue
            for f in sorted(d.glob("*.yaml")):
                try:
                    p = from_dict(yaml.safe_load(f.read_text()))
                    if p.fingerprint:
                        self._mem[variant_key(p.fingerprint, p)] = p
                except Exception:
                    pass

    def candidates(self, fp: str) -> list[Profile]:
        """All cached profiles for this fingerprint, richest (most levels) first."""
        out = [p for k, p in self._mem.items() if p.fingerprint == fp]
        return sorted(out, key=lambda p: len(p.levels), reverse=True)

    def get(self, fp: str) -> Profile | None:
        c = self.candidates(fp)
        return c[0] if c else None

    def put(self, fp: str, profile: Profile) -> None:
        """Store a profile as a depth-VARIANT keyed by fingerprint + levels-found,
        so e.g. ПКУ's стаття+розділ+підрозділ variant coexists with ККУ's
        стаття+розділ variant under the same fingerprint."""
        import yaml
        profile.fingerprint = fp
        key = variant_key(fp, profile)
        self._mem[key] = profile
        safe = "".join(c if c.isalnum() else "_" for c in key) or "none"
        (self.cache_dir / f"{safe}.yaml").write_text(
            yaml.safe_dump(to_dict(profile), allow_unicode=True))


def variant_key(fp: str, profile: Profile) -> str:
    return f"{fp}::{'+'.join(l.name for l in profile.levels)}"
