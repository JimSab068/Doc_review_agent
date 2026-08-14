"""
Stage 3.3: Tokenization vault.

This is the privacy boundary of the whole system. It is the ONLY component
permitted to hold the raw-value <-> token mapping. Everything downstream
of tokenize_document() operates on tokens only.

Two backends are supported, selected via VAULT_BACKEND ("memory" | "dynamodb"):

  memory    -- in-process dict, Fernet-encrypted with a locally held key.
               Used for local dev and the existing pytest suite.
  dynamodb  -- persistent, KMS-encrypted store in DynamoDB. Used in AWS so
               that a token map survives process restarts and is readable
               by any ECS task holding the correct IAM role -- not just the
               task that created it.

The public API (VaultClient) is identical regardless of backend.

Fail-closed: if APP_ENV is "staging" or "production", VAULT_BACKEND must be
"dynamodb" -- construction raises RuntimeError otherwise, rather than
silently falling back to an in-memory vault that doesn't persist across
task restarts or across the multiple tasks a load balancer will route to.

Retention model: DynamoDB TTL (VAULT_TTL_SECONDS) is a defense-in-depth
fallback for token maps abandoned by a crashed request. It is NOT the
mechanism you should rely on for timely deletion -- discard() is the real
data-deletion operation and must be called (see VaultClient.discard) as
soon as a request's token map is no longer needed.

Architectural note: this design satisfies "PII never reaches an external
LLM API" and "PII is encrypted at rest, readable only via scoped IAM."
It does NOT achieve strict process isolation -- detokenize_text() and
assert_no_pii_leak() still decrypt raw PII inside this same application
process, not a separate vault service. Describe it accurately as an
encrypted, persistent, IAM-scoped vault -- not as PII being kept out of
this process's memory space entirely. True process isolation would require
splitting this into a standalone internal vault API.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet, InvalidToken

from src.schemas import Document, Page, PIISpan, TokenizedDocument


class VaultSecurityError(Exception):
    """Raised when a raw PII value is detected somewhere it must never be --
    e.g. in a payload about to be sent to an external LLM API -- or when a
    value can't be safely stored (e.g. exceeds KMS's direct-encrypt limit)."""


def _make_token(field_type: str) -> str:
    return f"[[PII_{field_type.upper()}_{uuid.uuid4().hex[:8]}]]"


class _EncryptedStore:
    """Encrypted key-value store: token -> raw value, scoped by map_id.

    In-memory dict. Used for VAULT_BACKEND=memory (local dev, unit tests).
    The encryption key itself should come from a secrets manager in any
    deployed environment -- never hardcoded.
    """

    def __init__(self, encryption_key: bytes | None = None):
        self._fernet = Fernet(encryption_key or Fernet.generate_key())
        self._store: dict[str, dict[str, bytes]] = {}  # map_id -> {token: ciphertext}

    def put(self, map_id: str, token: str, raw_value: str) -> None:
        self._store.setdefault(map_id, {})[token] = self._fernet.encrypt(raw_value.encode())

    def get(self, map_id: str, token: str) -> str:
        try:
            ciphertext = self._store[map_id][token]
        except KeyError as exc:
            raise KeyError(f"Unknown token '{token}' for map '{map_id}'") from exc
        try:
            return self._fernet.decrypt(ciphertext).decode()
        except InvalidToken as exc:
            raise VaultSecurityError("Vault entry failed decryption integrity check") from exc

    def raw_values_for_map(self, map_id: str) -> list[str]:
        """Decrypt and return every raw value stored for a map. Used only
        internally by leak-detection assertions -- never exposed to agent
        code or logged."""
        return [
            self._fernet.decrypt(ciphertext).decode()
            for ciphertext in self._store.get(map_id, {}).values()
        ]

    def clear(self, map_id: str) -> None:
        self._store.pop(map_id, None)


class _DynamoDBStore:
    """Persistent, KMS-encrypted key-value store: token -> raw value, scoped
    by map_id. Backs VaultClient when VAULT_BACKEND=dynamodb.

    Each token's raw value is encrypted individually via AWS KMS direct
    Encrypt/Decrypt (values here are short PII strings -- names, SSNs,
    account numbers -- so no envelope/data-key management is needed).
    Ciphertext blobs live in a single DynamoDB item per token_map_id, under
    a 'tokens' map attribute keyed by token.

    Every KMS call passes an EncryptionContext binding the ciphertext to
    this table + this map_id + this token. KMS refuses to decrypt unless
    the *exact same* context is supplied. This means a ciphertext copied
    from one request's map into another (whether by bug or by an attacker
    with DynamoDB read access) fails to decrypt even though the KMS key
    itself would technically be able to -- the context, not just the key,
    has to match.

    KMS's direct Encrypt/Decrypt API caps plaintext at 4KB. put() enforces
    this explicitly rather than letting a KMS API error surface deep in a
    pipeline run.

    Requires (see create_vault_infra.py for provisioning):
      - A DynamoDB table with partition key 'token_map_id' (string) and a
        TTL attribute named 'expires_at'.
      - A customer-managed KMS key.
      - An IAM role/policy scoping the caller to exactly this table and key
        (see create_vault_infra.py, which generates the filled-in policy).

    Never logs raw values, ciphertext, or KMS plaintext. Callers must not
    either -- don't str()/repr() items pulled from this store outside of
    get()/raw_values_for_map()'s return values.
    """

    _MAX_KMS_PLAINTEXT_BYTES = 4096  # KMS Encrypt/Decrypt direct-call limit

    def __init__(
        self,
        table_name: str | None = None,
        kms_key_id: str | None = None,
        region_name: str | None = None,
        ttl_seconds: int | None = None,
    ):
        import boto3  # deferred import so VAULT_BACKEND=memory never requires boto3

        self._table_name = table_name or os.environ["VAULT_DYNAMODB_TABLE"]
        self._kms_key_id = kms_key_id or os.environ["VAULT_KMS_KEY_ID"]

        ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else int(os.environ.get("VAULT_TTL_SECONDS", "3600"))
        )
        if ttl_seconds <= 0:
            raise ValueError(
                f"VAULT_TTL_SECONDS must be a positive number of seconds (fallback cleanup "
                f"window for abandoned token maps); got {ttl_seconds}. Document your chosen "
                f"retention window when setting this."
            )
        self._ttl_seconds = ttl_seconds

        session = boto3.session.Session(region_name=region_name or os.environ.get("AWS_REGION"))
        self._table = session.resource("dynamodb").Table(self._table_name)
        self._kms = session.client("kms")

    # -- encryption helpers ---------------------------------------------

    def _encryption_context(self, map_id: str, token: str) -> dict[str, str]:
        return {"table": self._table_name, "token_map_id": map_id, "token": token}

    def _encrypt(self, map_id: str, token: str, raw_value: str) -> bytes:
        plaintext = raw_value.encode("utf-8")
        if len(plaintext) > self._MAX_KMS_PLAINTEXT_BYTES:
            raise VaultSecurityError(
                f"PII value for token '{token}' exceeds KMS's "
                f"{self._MAX_KMS_PLAINTEXT_BYTES}-byte direct-encryption limit; refusing to "
                f"store (value redacted from this error message)."
            )
        resp = self._kms.encrypt(
            KeyId=self._kms_key_id,
            Plaintext=plaintext,
            EncryptionContext=self._encryption_context(map_id, token),
        )
        return resp["CiphertextBlob"]

    def _decrypt(self, map_id: str, token: str, ciphertext: bytes) -> str:
        resp = self._kms.decrypt(
            KeyId=self._kms_key_id,
            CiphertextBlob=bytes(ciphertext),
            EncryptionContext=self._encryption_context(map_id, token),
        )
        return resp["Plaintext"].decode("utf-8")

    # -- public interface (mirrors _EncryptedStore) ----------------------

    def put(self, map_id: str, token: str, raw_value: str) -> None:
        ciphertext = self._encrypt(map_id, token, raw_value)
        now = datetime.now(timezone.utc)
        expires_at = int((now + timedelta(seconds=self._ttl_seconds)).timestamp())

        # Step 1: ensure the item and its 'tokens' map exist (no-op on
        # every call after the first for a given map_id).
        self._table.update_item(
            Key={"token_map_id": map_id},
            UpdateExpression=(
                "SET tokens = if_not_exists(tokens, :empty), "
                "created_at = if_not_exists(created_at, :now), "
                "expires_at = if_not_exists(expires_at, :exp)"
            ),
            ExpressionAttributeValues={":empty": {}, ":now": now.isoformat(), ":exp": expires_at},
        )
        # Step 2: write this token's ciphertext into the map.
        self._table.update_item(
            Key={"token_map_id": map_id},
            UpdateExpression="SET tokens.#tok = :ct",
            ExpressionAttributeNames={"#tok": token},
            ExpressionAttributeValues={":ct": ciphertext},
        )

    def get(self, map_id: str, token: str) -> str:
        resp = self._table.get_item(Key={"token_map_id": map_id})
        item = resp.get("Item")
        if not item or token not in item.get("tokens", {}):
            raise KeyError(f"Unknown token '{token}' for map '{map_id}'")
        return self._decrypt(map_id, token, item["tokens"][token])

    def raw_values_for_map(self, map_id: str) -> list[str]:
        """Decrypt and return every raw value stored for a map. Used only
        internally by leak-detection assertions -- never exposed to agent
        code or logged."""
        resp = self._table.get_item(Key={"token_map_id": map_id})
        item = resp.get("Item")
        if not item:
            return []
        return [self._decrypt(map_id, tok, ct) for tok, ct in item.get("tokens", {}).items()]

    def clear(self, map_id: str) -> None:
        self._table.delete_item(Key={"token_map_id": map_id})


def _build_store(encryption_key: bytes | None, backend: str | None):
    backend = backend or os.environ.get("VAULT_BACKEND", "memory")

    app_env = os.environ.get("APP_ENV", "").lower()
    if app_env in {"staging", "production"} and backend != "dynamodb":
        raise RuntimeError(
            f"APP_ENV='{app_env}' but VAULT_BACKEND='{backend}'. Refusing to start with an "
            f"in-memory vault in a deployed environment -- it doesn't persist across restarts "
            f"or across the multiple tasks a load balancer routes to. Set "
            f"VAULT_BACKEND=dynamodb explicitly."
        )

    if backend == "memory":
        return _EncryptedStore(encryption_key)
    if backend == "dynamodb":
        return _DynamoDBStore()
    raise ValueError(f"Unknown VAULT_BACKEND '{backend}' (expected 'memory' or 'dynamodb')")


class VaultClient:
    """Public interface every other component uses to interact with the
    vault. No caller outside this module ever sees an encryption key,
    KMS key material, or the raw store."""

    def __init__(self, encryption_key: bytes | None = None, backend: str | None = None):
        self._store = _build_store(encryption_key, backend)

    def tokenize_document(self, document: Document, spans: list[PIISpan]) -> TokenizedDocument:
        """Replace every PIISpan's raw text with an opaque token. Returns a
        TokenizedDocument plus a token_map_id that scopes this request's
        mapping in the vault.

        Callers MUST wrap all downstream processing (LLM calls, audit
        writes, everything) in a try/finally that calls discard() on this
        token_map_id in the finally block -- see module docstring on
        retention model. TTL is a fallback, not a substitute for this."""
        map_id = str(uuid.uuid4())

        spans_by_page: dict[int, list[PIISpan]] = {}
        for span in spans:
            spans_by_page.setdefault(span.page_number, []).append(span)

        new_pages: list[Page] = []
        for page in document.pages:
            text = page.text
            page_spans = sorted(
                spans_by_page.get(page.page_number, []),
                key=lambda s: s.start_char,
                reverse=True,  # replace back-to-front so earlier offsets stay valid
            )
            for span in page_spans:
                token = _make_token(span.field_type.value)
                self._store.put(map_id, token, span.raw_value)
                text = text[: span.start_char] + token + text[span.end_char :]
            new_pages.append(Page(page_number=page.page_number, text=text))

        return TokenizedDocument(
            doc_id=document.doc_id,
            doc_type=document.doc_type,
            pages=new_pages,
            token_map_id=map_id,
        )

    def detokenize_text(self, token_map_id: str, text: str) -> str:
        """Replace every token present in `text` with its raw value. Used
        only at the final output stage (3.8), never mid-pipeline."""
        import re

        def _replace(match: re.Match) -> str:
            token = match.group(0)
            try:
                return self._store.get(token_map_id, token)
            except KeyError:
                return token  # not one of ours; leave untouched

        return re.sub(r"\[\[PII_[A-Z_]+_[0-9a-f]{8}\]\]", _replace, text)

    def assert_no_pii_leak(self, token_map_id: str, payload: str) -> None:
        """Raise VaultSecurityError if any raw value tokenized under this
        map appears verbatim in `payload`. Call this on every outbound
        request to an external API as a hard safety check -- this is the
        mechanism that lets you *prove*, not just claim, that no raw PII
        reached the LLM."""
        for raw_value in self._store.raw_values_for_map(token_map_id):
            if raw_value and raw_value in payload:
                raise VaultSecurityError(
                    f"Raw PII value detected in outbound payload (field leaked, "
                    f"value redacted from this error message)"
                )

    def discard(self, token_map_id: str) -> None:
        """Drop a request's token map entirely once it's no longer needed.
        This -- not DynamoDB TTL -- is the actual data-deletion operation.
        Call it from a finally block that wraps everything from right after
        tokenize_document() through the end of the request."""
        self._store.clear(token_map_id)