"""
Rate limiting for Tier B live tests.

The free-tier Gemini API key allows a low requests-per-minute ceiling
(commonly 10 RPM on flash-lite tiers, but this is configurable in case
your quota differs). Tier A never touches this concern since it never
makes a real call. Tier B does on every persona, so this module owns two
things:

1. RateLimiter -- a client-side sliding-window limiter that proactively
   spaces out calls so we never *attempt* to exceed the quota in the
   first place. This is the primary defense.
2. RateLimitedLLMClient -- wraps any LLMClient (real GeminiClient, or
   anything else conforming to the same .generate(prompt) -> str
   protocol used by PrimaryAgent) so the limiter is enforced
   transparently. It also retries with exponential backoff if the API
   itself still returns a 429/RESOURCE_EXHAUSTED error -- belt and
   suspenders, since client-side pacing and the server's own window
   can drift out of sync by a call or two.

Both pieces are synchronous / blocking (time.sleep, not asyncio.sleep).
This is intentional: PersonaTestHarness.run_batch already executes
personas strictly one at a time (a plain for-loop with await, no
gather/concurrency), so a blocking sleep here does not stall anything
that should have been running in parallel. If you ever parallelize
Tier B execution, swap this for an asyncio.Semaphore/asyncio.sleep
based limiter instead -- a blocking sleep would then stall the whole
event loop, not just the one persona.
"""

from __future__ import annotations

import threading
import time

from collections import deque
from typing import Callable, Optional


class RateLimiter:
    """Sliding-window rate limiter: allows at most `max_calls` calls in
    any rolling `period_seconds` window. Blocks (sleeps) rather than
    raising when the limit would be exceeded.

    Thread-safe: safe to share a single instance across a test session
    even though Tier B currently only calls it sequentially.
    """

    def __init__(self, max_calls: int = 10, period_seconds: float = 60.0):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._call_times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until it's safe to make another call, then record it."""
        with self._lock:
            now = time.monotonic()

            # Drop timestamps that have aged out of the window.
            while self._call_times and now - self._call_times[0] >= self.period_seconds:
                self._call_times.popleft()

            if len(self._call_times) >= self.max_calls:
                # Oldest call in the window determines when a slot frees up.
                sleep_for = self.period_seconds - (now - self._call_times[0])
                if sleep_for > 0:
                    # Release the lock while sleeping so we don't block
                    # any other thread's bookkeeping -- reacquire after.
                    self._lock.release()
                    try:
                        time.sleep(sleep_for)
                    finally:
                        self._lock.acquire()
                    now = time.monotonic()
                    while self._call_times and now - self._call_times[0] >= self.period_seconds:
                        self._call_times.popleft()

            self._call_times.append(time.monotonic())

    def current_load(self) -> int:
        """Number of calls counted in the current window. Useful for
        logging/debugging test runs, not required for correctness."""
        with self._lock:
            now = time.monotonic()
            while self._call_times and now - self._call_times[0] >= self.period_seconds:
                self._call_times.popleft()
            return len(self._call_times)


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    """Best-effort sniff of whether an exception represents a 429 /
    RESOURCE_EXHAUSTED response, without depending on google-genai's
    internal exception types (those can change between SDK versions).
    Matches on the LLMAPIError message text produced by
    safe_exception_message() in primary_agent.py's GeminiClient."""
    msg = str(exc).lower()
    return any(marker in msg for marker in ("429", "resource_exhausted", "rate limit", "quota"))


class RateLimitedLLMClient:
    """Wraps any object exposing .generate(prompt) -> str (GeminiClient,
    or any LLMClient) with:
      - proactive client-side pacing via RateLimiter
      - retry-with-backoff if the API still rejects a call as rate-limited

    Drops in wherever a raw LLMClient is expected -- PrimaryAgent only
    calls .generate(prompt), so this satisfies that protocol unchanged.
    """

    def __init__(
        self,
        inner_client,
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 3,
        backoff_seconds: Callable[[int], float] = lambda attempt: min(60.0, 5.0 * (2 ** attempt)),
        on_wait: Optional[Callable[[float], None]] = None,
    ):
        self._inner = inner_client
        self._limiter = rate_limiter or RateLimiter(max_calls=10, period_seconds=60.0)
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._on_wait = on_wait

        # Mirrors GeminiClient's attribute so SecureAuditPipeline's
        # model_name detection (getattr(llm_client, "_model_name", ...))
        # keeps working when this wrapper is passed in its place.
        self._model_name = getattr(inner_client, "_model_name", "gemini-live-tier-b")

    def generate(self, prompt: str) -> str:
        self._limiter.acquire()

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._inner.generate(prompt)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-raised below
                last_exc = exc
                if attempt >= self._max_retries or not _looks_like_rate_limit_error(exc):
                    raise
                wait = self._backoff_seconds(attempt)
                if self._on_wait:
                    self._on_wait(wait)
                time.sleep(wait)
                # Re-acquire a slot before retrying -- the failed attempt
                # may or may not have counted against the server's quota,
                # but treating it as if it did is the conservative choice.
                self._limiter.acquire()

        # Unreachable in practice (loop either returns or raises), but
        # keeps type-checkers happy and fails loudly if it's ever hit.
        raise last_exc  # type: ignore[misc]