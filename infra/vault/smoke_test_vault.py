"""
Real-AWS smoke test for the persistent vault -- run once against your
provisioned dev/staging resources before wiring the vault into the
FastAPI service.

This is deliberately NOT a pytest file and NOT mocked -- it exercises the
exact runtime path (VaultClient -> _DynamoDBStore -> real DynamoDB + real
KMS) using whatever AWS credentials are active in your shell. moto-based
tests already cover correctness in isolation; this exists to catch the
things moto can't: IAM permission gaps, region mismatches, real KMS
ciphertext behavior, real DynamoDB latency/consistency.

Prerequisites:
  1. The generated policy (infra/vault/vault_iam_policy.generated.json)
     must be attached to whatever IAM identity you run this as -- ideally
     as a SEPARATE inline policy on top of your existing provisioning
     permissions, so this test proves the runtime policy alone is
     sufficient (matching what the ECS task role will actually have).
  2. Environment variables set for the environment you provisioned, e.g.:
       APP_ENV=dev            (or omit -- APP_ENV isn't checked below)
       VAULT_BACKEND=dynamodb
       VAULT_DYNAMODB_TABLE=loan-kyc-token-maps-dev
       VAULT_KMS_KEY_ID=<the ARN printed when you ran create_vault_infra.py>
       AWS_REGION=us-east-1
       VAULT_TTL_SECONDS=3600

Usage (from repo root, with your venv active and those env vars set):
    python infra/vault/smoke_test_vault.py

Exits 0 and prints "SMOKE TEST PASSED" on success. Any failed assertion
raises and exits non-zero -- treat that as a real problem, not something
to silently retry past.
"""

from __future__ import annotations

import os
import sys

import boto3

# Make src/ importable when running this script directly from infra/vault/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ingestion import ingest_text
from src.pii_detector import RegexPIIDetector
from src.schemas import DocumentType
from src.vault import VaultClient

REQUIRED_ENV_VARS = [
    "VAULT_BACKEND",
    "VAULT_DYNAMODB_TABLE",
    "VAULT_KMS_KEY_ID",
    "AWS_REGION",
]

SAMPLE_TEXT = "Applicant: Jane Q. Smoketest, SSN: 900-11-2222"
SYNTHETIC_SSN = "900-11-2222"


def _check_env():
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Set these to your dev/staging values before running this script.")
        sys.exit(1)
    if os.environ["VAULT_BACKEND"] != "dynamodb":
        print(f"VAULT_BACKEND is '{os.environ['VAULT_BACKEND']}', expected 'dynamodb'.")
        print("This smoke test is specifically for the persistent backend.")
        sys.exit(1)


def main():
    _check_env()
    table_name = os.environ["VAULT_DYNAMODB_TABLE"]
    region = os.environ["AWS_REGION"]

    print(f"Target table:  {table_name}")
    print(f"Target region: {region}")
    print(f"KMS key:       {os.environ['VAULT_KMS_KEY_ID']}")
    print()

    # -- Step 1: tokenize a synthetic document through the real path --------
    print("[1/5] Ingesting synthetic document and detecting PII...")
    document = ingest_text(SAMPLE_TEXT, DocumentType.LOAN_APPLICATION, "smoketest.txt")
    spans = RegexPIIDetector().detect(document)
    assert spans, "Detector found no PII spans -- check RegexPIIDetector against SAMPLE_TEXT"
    print(f"      Detected {len(spans)} span(s): {[s.field_type.value for s in spans]}")

    vault = VaultClient()  # reads VAULT_BACKEND/VAULT_DYNAMODB_TABLE/VAULT_KMS_KEY_ID from env
    tokenized = vault.tokenize_document(document, spans)
    assert SYNTHETIC_SSN not in tokenized.pages[0].text, "Raw SSN leaked into tokenized text"
    print(f"      token_map_id = {tokenized.token_map_id}")
    print(f"      Tokenized text: {tokenized.pages[0].text}")

    # -- Step 2: confirm DynamoDB holds ciphertext, not plaintext -----------
    print("\n[2/5] Reading raw item back from DynamoDB directly...")
    ddb = boto3.resource("dynamodb", region_name=region)
    item = ddb.Table(table_name).get_item(Key={"token_map_id": tokenized.token_map_id}).get("Item")
    assert item is not None, "Token map not found in DynamoDB -- write appears to have failed"
    tokens_blob = item.get("tokens", {})
    assert tokens_blob, "Item exists but has no 'tokens' map"
    for token, ciphertext in tokens_blob.items():
        raw_bytes = bytes(ciphertext)
        assert SYNTHETIC_SSN.encode() not in raw_bytes, (
            f"PLAINTEXT SSN FOUND IN DYNAMODB under token {token} -- KMS encryption did not happen"
        )
    print(f"      Confirmed {len(tokens_blob)} ciphertext blob(s) in DynamoDB, no plaintext present")

    # -- Step 3: fresh VaultClient instance detokenizes correctly -----------
    # A second instance simulates a different ECS task/process reading a
    # token map it didn't write -- this is the actual scenario the whole
    # persistent-vault migration exists to support.
    print("\n[3/5] Creating a fresh VaultClient instance and detokenizing...")
    vault_reader = VaultClient()
    restored = vault_reader.detokenize_text(tokenized.token_map_id, tokenized.pages[0].text)
    assert restored == document.pages[0].text, (
        f"Detokenized text does not match original.\n  expected: {document.pages[0].text!r}\n  got:      {restored!r}"
    )
    print(f"      Restored: {restored}")

    # -- Step 4: discard the token map -----------------------------------
    print("\n[4/5] Discarding token map...")
    vault_reader.discard(tokenized.token_map_id)

    # -- Step 5: confirm the map is actually gone ----------------------
    print("\n[5/5] Confirming second lookup fails after discard...")
    post_discard_item = ddb.Table(table_name).get_item(Key={"token_map_id": tokenized.token_map_id}).get("Item")
    assert post_discard_item is None, "Item still present in DynamoDB after discard() -- deletion did not happen"

    # detokenize_text() leaves unresolvable tokens as-is rather than raising
    # (matches the memory backend's behavior) -- confirm that's what happens
    # now that the underlying data is gone.
    post_discard_result = vault_reader.detokenize_text(tokenized.token_map_id, tokenized.pages[0].text)
    assert post_discard_result == tokenized.pages[0].text, (
        "Expected tokens to remain unresolved (left as literal token text) after discard"
    )
    print("      Confirmed: item deleted from DynamoDB, token no longer resolves")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()