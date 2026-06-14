"""Environment-driven settings for ScribblesLM v2.

Loaded lazily so the server starts cleanly even with missing keys; each backend
validates the keys it actually needs on first use (per the error-handling spec).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def _bool(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # storage
    db_path: Path
    models_dir: Path

    # public path — Voyage
    voyage_api_key: str | None
    voyage_model: str            # default voyage-context-3
    voyage_fallback_model: str   # default voyage-4-lite ($0.02/M + 200M free; NOT 3.5)
    voyage_concurrency: int      # R1 seam: default 1
    voyage_tpm_ceiling: int      # R1 governor: default 3,000,000 (context-3 Tier-1)

    # private path — local bge-m3 GGUF
    private_model_path: Path
    private_threads: int

    # private-path contextualization (DeepSeek / OpenAI-compatible)
    context_llm_base_url: str | None
    context_llm_api_key: str | None
    context_llm_model: str | None

    # routing default
    default_private: bool

    # induction (Milestone D)
    profile_cache_dir: Path
    profile_synthesis: bool   # rung-4 LLM profile synthesis (default on)

    # security
    documents_root: Path | None   # if set, local file paths must resolve inside this dir
    max_document_bytes: int       # 0 = unlimited (not recommended)

    # reranker (Milestone C) — DEFAULT OFF (token cost + latency)
    reranker_enabled: bool
    reranker_model: str          # local private reranker (self-exiting subprocess)
    reranker_idle_exit: int      # seconds of idle before the subprocess exits
    voyage_rerank_model: str     # public reranker (Voyage API)


def _load_dotenv() -> None:
    """Populate os.environ from a .env file (cwd or ~/.scribbleslm), without
    overriding values already set in the real environment (MCP-config env wins)."""
    for cand in (Path.cwd() / ".env", _expand("~/.scribbleslm/.env")):
        try:
            if cand.is_file():
                for line in cand.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        except OSError:
            pass


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    home = _expand(os.environ.get("SCRIBBLESLM_HOME", "~/.scribbleslm"))
    return Settings(
        db_path=_expand(os.environ.get("SCRIBBLESLM_DB_PATH", str(home / "scribbleslm.db"))),
        models_dir=_expand(os.environ.get("SCRIBBLESLM_MODELS_DIR", str(home / "models"))),
        voyage_api_key=(os.environ.get("VOYAGE_API_KEY") or "").strip() or None,
        voyage_model=os.environ.get("VOYAGE_MODEL", "voyage-context-3"),
        voyage_fallback_model=os.environ.get("VOYAGE_FALLBACK_MODEL", "voyage-4-lite"),
        voyage_concurrency=int(os.environ.get("VOYAGE_CONCURRENCY", "1")),
        voyage_tpm_ceiling=int(os.environ.get("VOYAGE_TPM_CEILING", "3000000")),
        private_model_path=_expand(
            os.environ.get("PRIVATE_EMBEDDING_MODEL_PATH", str(home / "models" / "bge-m3-Q5_K_M.gguf"))
        ),
        private_threads=int(os.environ.get("PRIVATE_EMBEDDING_THREADS", "4")),
        context_llm_base_url=os.environ.get("CONTEXT_LLM_BASE_URL"),
        context_llm_api_key=os.environ.get("CONTEXT_LLM_API_KEY"),
        context_llm_model=os.environ.get("CONTEXT_LLM_MODEL"),
        default_private=_bool(os.environ.get("DEFAULT_PRIVATE"), False),
        profile_cache_dir=_expand(os.environ.get("PROFILE_CACHE_DIR", str(home / "profiles"))),
        profile_synthesis=_bool(os.environ.get("PROFILE_SYNTHESIS"), True),
        documents_root=_expand(os.environ["SCRIBBLESLM_DOCUMENTS_ROOT"])
            if os.environ.get("SCRIBBLESLM_DOCUMENTS_ROOT") else None,
        max_document_bytes=int(os.environ.get("SCRIBBLESLM_MAX_DOCUMENT_BYTES", str(50 * 1024 * 1024))),
        reranker_enabled=_bool(os.environ.get("RERANKER_ENABLED"), False),
        reranker_model=os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_idle_exit=int(os.environ.get("RERANKER_IDLE_EXIT", "120")),
        voyage_rerank_model=os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5"),
    )


def reset_settings_cache() -> None:
    """Test helper — re-read env on next get_settings()."""
    get_settings.cache_clear()
