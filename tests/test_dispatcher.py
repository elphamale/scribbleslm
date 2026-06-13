import asyncio

import pytest

from scribbleslm.embeddings.dispatcher import Dispatcher, TokenGovernor, is_rate_limit


def test_is_rate_limit_detection():
    assert is_rate_limit(Exception("HTTP 429 Too Many Requests"))
    assert is_rate_limit(type("RateLimitError", (Exception,), {})("x"))
    assert is_rate_limit(Exception("rate limit exceeded"))
    assert not is_rate_limit(Exception("connection reset"))


async def test_retries_on_rate_limit_then_succeeds():
    d = Dispatcher(wait_initial=0.001, wait_max=0.01, max_retries=5)
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("429 rate limit")
        return "ok"

    assert await d.run(fn, tokens=10) == "ok"
    assert calls["n"] == 3


async def test_no_retry_on_non_rate_error():
    d = Dispatcher(wait_initial=0.001)
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await d.run(fn)
    assert calls["n"] == 1


def test_governor_tracks_tokens():
    g = TokenGovernor(1000)
    g.record(400)
    g.record(300)
    assert g.current_tpm == 700


async def test_semaphore_limits_concurrency():
    d = Dispatcher(concurrency=1, wait_initial=0.001)
    active = {"now": 0, "max": 0}

    async def fn():
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0.02)
        active["now"] -= 1
        return 1

    await asyncio.gather(*(d.run(fn) for _ in range(4)))
    assert active["max"] == 1  # concurrency=1 -> never overlapped
