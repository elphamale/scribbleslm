"""C-3: the local reranker is a SELF-EXITING SUBPROCESS — zero idle footprint.

Uses RERANKER_MODEL=stub (stdlib-only worker) so the lifecycle is verified without
downloading the cross-encoder. The key assertion: the subprocess exits after idle.
"""
import time

from scribbleslm.config import get_settings, reset_settings_cache
from scribbleslm.reranker import LocalReranker, Reranker


def _settings(monkeypatch, idle="1"):
    monkeypatch.setenv("RERANKER_MODEL", "stub")
    monkeypatch.setenv("RERANKER_IDLE_EXIT", idle)
    monkeypatch.setenv("RERANKER_ENABLED", "1")
    reset_settings_cache()
    return get_settings()


def test_spawn_on_use_score_then_self_exit(monkeypatch):
    lr = LocalReranker(_settings(monkeypatch, idle="1"))
    assert not lr.is_running()                       # nothing resident before first use
    scores = lr.score("податок на прибуток", ["текст про податок", "geological survey"])
    assert lr.is_running()                           # spawned on first rerank
    assert len(scores) == 2 and scores[0] > scores[1]
    time.sleep(1.8)                                  # exceed idle timeout
    assert not lr.is_running()                       # SELF-EXITED — the zero-idle guarantee
    lr.close()


def test_respawns_transparently_after_idle_exit(monkeypatch):
    lr = LocalReranker(_settings(monkeypatch, idle="1"))
    lr.score("a", ["a b"])
    time.sleep(1.8)
    assert not lr.is_running()
    scores = lr.score("a", ["a b", "c d"])           # respawns under the hood
    assert lr.is_running() and len(scores) == 2
    lr.close()


def test_route_is_sensitivity_aware(monkeypatch):
    r = Reranker(_settings(monkeypatch))
    assert r.route(any_private=False, query_private=False) == "voyage"
    assert r.route(any_private=True, query_private=False) == "local"   # private candidate
    assert r.route(any_private=False, query_private=True) == "local"   # sensitive query


async def test_private_rerank_routes_to_local_subprocess(monkeypatch):
    r = Reranker(_settings(monkeypatch))
    scores = await r.rerank("податок", [("про податок", True), ("інше", True)], query_private=True)
    assert len(scores) == 2 and scores[0] >= scores[1]
    r.local.close()
