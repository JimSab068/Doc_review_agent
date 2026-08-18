"""
Production rate limiting for outbound Gemini calls (generate + embeddings).

Mirrors tests/tier_b_live/rate_limiter.py's design (blocking time.sleep,
thread-safe via threading.Lock), moved into src/ so it actually ships in
the Docker image -- tests/ is never copied into the container (see
Dockerfile's `COPY src/ ./src/`), so api.py could not have imported the
test version even if the design were otherwise fine to reuse as-is.

IMPORTANT -- why a blocking time.sleep is safe here and NOT a bug:
api.py is an async FastAPI app serving concurrent requests over one
event loop. If RateLimiter.acquire() were called directly from an
`async def` request handler, one request's rate-limit wait would freeze
every other in-flight request (including /healthz) for the duration of
the sleep. That is NOT what happens here: pipeline.py already offloads
the primary agent, critic agent, and KB calls via asyncio.to_thread (see
its history -- this fix predates this file). RateLimitedLLMClient and
RateLimitedEmbeddingFunction below are only ever invoked from *inside*
those to_thread-offloaded calls, i.e. on a worker thread, never directly
on the event loop. A blocking sleep on a worker thread only blocks that
thread -- other concurrent requests' worker threads, and the event loop
itself, are unaffected. If pipeline.py's to_thread offloading is ever
removed, this design must be revisited (swap to an asyncio.Lock +
asyncio.sleep based limiter instead).

Net effect under concurrent traffic: requests contend for the same
shared budget and queue up (each blocked thread waits its turn) rather
than the service exceeding Gemini's RPM and getting 429s. Set
GEMINI_MAX_CALLS_PER_MINUTE to your actual Gemini plan's limit -- see
config.py's gemini_max_calls_per_minute field.
"""

from __future__ import annotations

import threading
import time

from collections import deque
from typing import Callable, Optional


class RateLimiter:
    """Sliding-window rate limiter: allows at most `max_calls` calls in
    any rolling `period_seconds` window. Blocks (sleeps) rather than
    raising when the limit would be exceeded. Thread-safe -- see module
    docstring for why blocking here is safe under FastAPI specifically."""

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

            while self._call_times and now - self._call_times[0] >= self.period_seconds:
                self._call_times.popleft()

            if len(self._call_times) >= self.max_calls:
                sleep_for = self.period_seconds - (now - self._call_times[0])
                if sleep_for > 0:
                    # Release the lock while sleeping so other threads'
                    # acquire() calls aren't blocked from even checking
                    # the window -- only this thread waits.
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
        """Number of calls counted in the current window -- exposed so
        /readyz or a future /debug endpoint can report real-time load."""
        with self._lock:
            now = time.monotonic()
            while self._call_times and now - self._call_times[0] >= self.period_seconds:
                self._call_times.popleft()
            return len(self._call_times)


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    """Best-effort sniff of a 429 / RESOURCE_EXHAUSTED response, without
    depending on google-genai's internal exception types."""
    msg = str(exc).lower()
    return any(marker in msg for marker in ("429", "resource_exhausted", "rate limit", "quota"))


class RateLimitedLLMClient:
    """Wraps any object exposing .generate(prompt) -> str (GeminiClient)
    with proactive client-side pacing plus retry-with-backoff if the API
    still rejects a call as rate-limited. Drops in wherever a raw
    LLMClient is expected -- PrimaryAgent/CriticAgent only call
    .generate(prompt), so this satisfies that protocol unchanged.

    generate() itself is still synchronous/blocking, same as the
    GeminiClient it wraps -- pipeline.py's existing asyncio.to_thread
    offload is what keeps this off the event loop, not anything in this
    class. Do not call this directly from async code."""

    def __init__(
        self,
        inner_client,
        rate_limiter: RateLimiter,
        max_retries: int = 3,
        backoff_seconds: Callable[[int], float] = lambda attempt: min(30.0, 2.0 * (2 ** attempt)),
        on_wait: Optional[Callable[[float], None]] = None,
    ):
        self._inner = inner_client
        self._limiter = rate_limiter
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._on_wait = on_wait
        self._model_name = getattr(inner_client, "_model_name", "gemini-rate-limited")

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
                self._limiter.acquire()

        raise last_exc  # type: ignore[misc]

    def __getattr__(self, name):
        # Forward anything else (e.g. any attribute PrimaryAgent/CriticAgent
        # might introspect off the client) to the wrapped instance.
        return getattr(self._inner, name)


class RateLimitedEmbeddingFunction:
    """Wraps a Chroma embedding function so each __call__ counts as one
    slot against a SHARED limiter -- Chroma's GoogleGeminiEmbeddingFunction
    calls the Gemini embeddings endpoint on the same API key/quota as
    generate() calls, so both need to draw against the same budget, not
    two independent ones whose sum can still blow past quota even if each
    individually looks fine. Mirrors live_clients.py's test-only version;
    this is the production copy."""

    def __init__(self, inner_embedding_function, rate_limiter: RateLimiter):
        self._inner = inner_embedding_function
        self._limiter = rate_limiter

    def __call__(self, input):  # noqa: A002 -- name required by Chroma's EmbeddingFunction protocol
        self._limiter.acquire()
        return self._inner(input)

    def __getattr__(self, name):
        return getattr(self._inner, name)