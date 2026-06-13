"""Async dispatch seam for remote embedding calls (R1).

Builds the *seam* for concurrency without yet running anything in parallel:
- asyncio.Semaphore (default concurrency=1) — flip the config to parallelize later
- tenacity exponential-backoff-with-jitter on HTTP 429 / rate-limit errors
- a token-budget governor STUB that tracks tokens/min against a ceiling

Do NOT add parallel execution logic here yet (Milestone B). The point is that
retrofitting concurrency into a serial dispatcher is a rewrite; the seam is cheap
to build now and is a one-line config flip later.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger("scribbleslm.dispatcher")
T = TypeVar("T")


def is_rate_limit(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "ratelimit" in name or "429" in msg or "rate limit" in msg or "too many requests" in msg


class TokenGovernor:
    """STUB: tracks tokens/min against a ceiling. Records and warns; does NOT
    throttle yet (real pacing is a Milestone B concern once concurrency > 1)."""

    def __init__(self, tpm_ceiling: int):
        self.tpm_ceiling = tpm_ceiling
        self._window: deque[tuple[float, int]] = deque()

    def record(self, tokens: int) -> None:
        now = time.monotonic()
        self._window.append((now, tokens))
        while self._window and now - self._window[0][0] > 60.0:
            self._window.popleft()
        if self.current_tpm > self.tpm_ceiling:
            log.warning(
                "token-budget governor: %d tok/min exceeds ceiling %d (not throttling yet)",
                self.current_tpm, self.tpm_ceiling,
            )

    @property
    def current_tpm(self) -> int:
        return sum(t for _, t in self._window)


class Dispatcher:
    def __init__(self, concurrency: int = 1, tpm_ceiling: int = 3_000_000,
                 max_retries: int = 6, wait_initial: float = 1.0, wait_max: float = 60.0):
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self.governor = TokenGovernor(tpm_ceiling)
        self.max_retries = max_retries
        self._wait_initial = wait_initial
        self._wait_max = wait_max

    async def run(self, fn: Callable[[], Awaitable[T]], *, tokens: int = 0) -> T:
        """Run one async API call under the semaphore, with 429 backoff and
        token accounting. `fn` must be an idempotent coroutine factory call."""
        async with self._sem:
            self.governor.record(tokens)
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(is_rate_limit),
                wait=wait_exponential_jitter(initial=self._wait_initial, max=self._wait_max),
                stop=stop_after_attempt(self.max_retries),
                reraise=True,
            ):
                with attempt:
                    return await fn()
            raise RuntimeError("unreachable")  # pragma: no cover
