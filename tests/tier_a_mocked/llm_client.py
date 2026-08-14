"""
Deterministic LLMClient stub for Tier A.

Never calls a real model, costs nothing, and never flakes -- its only
job is to prove the pipeline's wiring (tokenization, prompt construction,
vault gating, parsing, routing, audit write) is correct, independent of
whether any real model reasons well. Real LLM-quality metrics (extraction
accuracy, hallucination rate, injection catch rate, etc.) belong to Tier
B -- this stub cannot measure them, because it never actually reasons
about the prompt content at all.
"""

from __future__ import annotations

import re
from typing import Callable, List, Union

_TOKEN_PATTERN = re.compile(r"\[\[PII_([A-Z_]+)_[0-9a-f]{8}\]\]")

ScriptedResponse = Union[str, Callable[[str], str]]


def extract_tokens_by_type(prompt: str) -> dict[str, str]:
    """Return {pii_type_upper: first_matching_token} for every PII token
    type present in `prompt`. Used by response builders that need to echo
    a specific run's actual (randomly-suffixed) token back in the
    scripted JSON response -- the token itself is only known at runtime,
    since the vault mints a fresh random suffix on every tokenize call.
    """
    found: dict[str, str] = {}
    for match in _TOKEN_PATTERN.finditer(prompt):
        pii_type = match.group(1)
        found.setdefault(pii_type, match.group(0))
    return found


class ScriptedLLMClient:
    """Consumes a fixed, ordered list of responses -- one per `generate`
    call. Each entry is either a raw JSON string (returned verbatim) or a
    callable `(prompt: str) -> str` invoked against that call's actual
    prompt, so a response can reference runtime-only content (e.g. a
    randomly-suffixed PII token) without knowing it in advance.

    Exhausting the list raises LookupError rather than returning a
    default -- an unscripted call means the test doesn't actually cover
    what it claims to, and that should fail loudly, not silently pass.
    """

    def __init__(self, responses: List[ScriptedResponse]):
        self._responses = list(responses)
        self.prompts_sent: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts_sent.append(prompt)
        if not self._responses:
            raise LookupError(
                "ScriptedLLMClient: no more scripted responses configured -- "
                f"received an unscripted call. Prompt (first 200 chars): "
                f"{prompt[:200]!r}"
            )
        response = self._responses.pop(0)
        return response(prompt) if callable(response) else response