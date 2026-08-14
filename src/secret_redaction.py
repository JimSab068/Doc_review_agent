"""
Shared helper for keeping credentials out of anything that might end up on
a terminal, in a log line, or inside a raised exception's message.

Rationale: SDKs occasionally echo request state -- including auth headers
or query strings -- back inside exception messages on network failures.
Never passing a key as a function argument (see GeminiClient / ComplianceKB,
which both read credentials directly from os.environ at the point of use,
not via a parameter that could get logged, repr'd, or passed down a call
stack) is the primary defense. This module is the backstop for the case
where a value leaks into error text anyway -- it is not a substitute for
that primary defense.
"""

from __future__ import annotations

import os

# Names of environment variables that may hold a credential. Add to this
# list any time a new component starts reading a secret from the
# environment -- it's the single place that needs updating.
_SECRET_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def redact_secrets(text: str) -> str:
    """Return `text` with any currently-set secret env var value replaced.

    Safe to call on text that contains no secrets -- it's a no-op in that
    case. Intentionally checks live env var values (not a fixed pattern),
    so it redacts whatever key is actually in use, not a guessed format.
    """
    if not text:
        return text
    redacted = text
    for var_name in _SECRET_ENV_VARS:
        value = os.environ.get(var_name)
        # Guard against redacting on a trivially short/empty value, which
        # could otherwise mangle unrelated text.
        if value and len(value) >= 8 and value in redacted:
            redacted = redacted.replace(value, f"[REDACTED:{var_name}]")
    return redacted


def safe_exception_message(exc: BaseException) -> str:
    """Redacted string form of an exception, safe to print or re-raise."""
    return redact_secrets(str(exc))