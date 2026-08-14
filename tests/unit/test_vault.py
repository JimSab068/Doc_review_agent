import pytest

from src.ingestion import ingest_text
from src.pii_detector import RegexPIIDetector
from src.schemas import DocumentType
from src.vault import VaultClient, VaultSecurityError

from cryptography.fernet import Fernet
from src.schemas import Document, Page, PIISpan, PIIType


class TestVaultClient:
    def setup_method(self):
        self.vault = VaultClient()
        self.detector = RegexPIIDetector()

    def test_tokenize_replaces_raw_pii_with_token(self):
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)

        tokenized = self.vault.tokenize_document(doc, spans)

        assert "123-45-6789" not in tokenized.pages[0].text
        assert "[[PII_SSN_" in tokenized.pages[0].text

    def test_tokenize_preserves_surrounding_text(self):
        doc = ingest_text("Applicant SSN: 123-45-6789 filed today.", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)

        tokenized = self.vault.tokenize_document(doc, spans)

        assert tokenized.pages[0].text.startswith("Applicant SSN: ")
        assert tokenized.pages[0].text.endswith(" filed today.")

    def test_tokenize_handles_multiple_spans_on_same_page(self):
        text = "SSN: 123-45-6789, Email: a@b.com, DOB: 01/01/1990"
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)

        tokenized = self.vault.tokenize_document(doc, spans)

        assert "123-45-6789" not in tokenized.pages[0].text
        assert "a@b.com" not in tokenized.pages[0].text
        assert "01/01/1990" not in tokenized.pages[0].text

    def test_detokenize_text_restores_original_values(self):
        text = "SSN: 123-45-6789, Email: a@b.com"
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)
        tokenized = self.vault.tokenize_document(doc, spans)

        restored = self.vault.detokenize_text(tokenized.token_map_id, tokenized.pages[0].text)

        assert restored == text

    def test_detokenize_leaves_unknown_tokens_untouched(self):
        result = self.vault.detokenize_text("nonexistent-map-id", "no tokens here")

        assert result == "no tokens here"

    def test_assert_no_pii_leak_passes_on_clean_payload(self):
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)
        tokenized = self.vault.tokenize_document(doc, spans)

        # Should not raise: the tokenized text has no raw PII in it
        self.vault.assert_no_pii_leak(tokenized.token_map_id, tokenized.pages[0].text)

    def test_assert_no_pii_leak_raises_when_raw_value_present(self):
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)
        tokenized = self.vault.tokenize_document(doc, spans)

        leaked_payload = "here is the raw SSN 123-45-6789 by mistake"

        with pytest.raises(VaultSecurityError):
            self.vault.assert_no_pii_leak(tokenized.token_map_id, leaked_payload)

    def test_assert_no_pii_leak_error_does_not_contain_raw_value(self):
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)
        tokenized = self.vault.tokenize_document(doc, spans)

        try:
            self.vault.assert_no_pii_leak(tokenized.token_map_id, "leak: 123-45-6789")
            assert False, "expected VaultSecurityError"
        except VaultSecurityError as exc:
            assert "123-45-6789" not in str(exc)

    def test_discard_removes_map_and_detokenize_no_longer_works(self):
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)
        tokenized = self.vault.tokenize_document(doc, spans)

        self.vault.discard(tokenized.token_map_id)
        restored = self.vault.detokenize_text(tokenized.token_map_id, tokenized.pages[0].text)

        # Tokens no longer resolve once discarded -- they pass through unchanged
        assert "[[PII_SSN_" in restored

    def test_each_tokenization_call_gets_a_distinct_map_id(self):
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)

        tok1 = self.vault.tokenize_document(doc, spans)
        tok2 = self.vault.tokenize_document(doc, spans)

        assert tok1.token_map_id != tok2.token_map_id

    def test_custom_encryption_key_initialization(self):
        """Verifies VaultClient works with an explicitly supplied Fernet encryption key."""
        custom_key = Fernet.generate_key()
        vault = VaultClient(encryption_key=custom_key)

        doc = ingest_text("SSN: 999-88-7777", DocumentType.LOAN_APPLICATION, "a.txt")
        detector = RegexPIIDetector()
        spans = detector.detect(doc)

        tokenized = vault.tokenize_document(doc, spans)
        restored = vault.detokenize_text(tokenized.token_map_id, tokenized.pages[0].text)

        assert restored == "SSN: 999-88-7777"

    def test_decryption_failure_raises_vault_security_error(self):
        """Triggers InvalidToken during decryption to exercise the VaultSecurityError branch in _EncryptedStore.get."""
        vault = VaultClient()
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        detector = RegexPIIDetector()
        spans = detector.detect(doc)

        tokenized = vault.tokenize_document(doc, spans)
        map_id = tokenized.token_map_id

        # Corrupt the encrypted ciphertext directly in internal store
        token = list(vault._store._store[map_id].keys())[0]
        vault._store._store[map_id][token] = b"corrupted_garbage_ciphertext"

        # Attempting to detokenize or get corrupt payload must raise VaultSecurityError
        with pytest.raises(VaultSecurityError, match="decryption integrity check"):
            vault._store.get(map_id, token)

    def test_multi_page_document_tokenization(self):
        """Verifies tokenization correctly groups spans across multiple document pages."""
        vault = VaultClient()
        doc = Document(
            doc_id="multi-page-doc",
            doc_type=DocumentType.LOAN_APPLICATION,
            source_filename="multi.txt",
            pages=[
                Page(page_number=1, text="Page 1 SSN: 111-11-1111"),
                Page(page_number=2, text="Page 2 Email: user@example.com"),
            ],
        )
        detector = RegexPIIDetector()
        spans = detector.detect(doc)

        tokenized = vault.tokenize_document(doc, spans)

        assert "111-11-1111" not in tokenized.pages[0].text
        assert "[[PII_SSN_" in tokenized.pages[0].text

        assert "user@example.com" not in tokenized.pages[1].text
        assert "[[PII_EMAIL_" in tokenized.pages[1].text

        # Verify full detokenization restores both pages
        restored_p1 = vault.detokenize_text(tokenized.token_map_id, tokenized.pages[0].text)
        restored_p2 = vault.detokenize_text(tokenized.token_map_id, tokenized.pages[1].text)

        assert restored_p1 == "Page 1 SSN: 111-11-1111"
        assert restored_p2 == "Page 2 Email: user@example.com"

    def test_assert_no_pii_leak_ignores_empty_raw_values(self):
        """Confirms that empty raw values in the store do not trigger false positive leak assertions."""
        vault = VaultClient()
        # Manually inject an empty string raw value into vault store
        vault._store.put("test_map", "[[PII_TEST_12345678]]", "")

        # Should not raise VaultSecurityError even though "" is in "any payload string"
        vault.assert_no_pii_leak("test_map", "Clean payload text")
    
