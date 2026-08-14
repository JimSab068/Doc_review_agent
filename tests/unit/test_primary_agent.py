import json

import pytest

from src.ingestion import ingest_text
from src.pii_detector import RegexPIIDetector
from src.primary_agent import LLMOutputParseError, PrimaryAgent, build_prompt
from src.schemas import DocumentType
from src.vault import VaultClient, VaultSecurityError


class StubLLMClient:
    """Records the prompt it was called with and returns a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.call_count += 1
        return self.response


VALID_RESPONSE = json.dumps(
    {
        "extracted_fields": {"requested_amount": "25000", "purpose": "home improvement"},
        "inconsistencies": [],
        "missing_compliance_items": [],
        "confidence": 0.92,
        "reasoning": "All fields present and consistent.",
    }
)


class TestBuildPrompt:
    def test_prompt_contains_tokenized_text_not_raw_pii(self):
        doc = ingest_text("SSN: 123-45-6789", DocumentType.LOAN_APPLICATION, "a.txt")
        spans = RegexPIIDetector().detect(doc)
        vault = VaultClient()
        tokenized = vault.tokenize_document(doc, spans)

        prompt = build_prompt(tokenized)

        assert "123-45-6789" not in prompt
        assert "[[PII_SSN_" in prompt


class TestPrimaryAgentRun:
    def setup_method(self):
        self.vault = VaultClient()
        self.detector = RegexPIIDetector()

    def _tokenized_doc(self, text="Requested amount: 25000, SSN: 123-45-6789"):
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")
        spans = self.detector.detect(doc)
        return self.vault.tokenize_document(doc, spans)

    def test_returns_draft_decision_on_valid_response(self):
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(VALID_RESPONSE)
        agent = PrimaryAgent(llm, self.vault, model_name="stub-model")

        result = agent.run(tokenized)

        assert result.doc_id == tokenized.doc_id
        assert result.confidence == 0.92
        assert result.extracted_fields["requested_amount"] == "25000"
        assert result.model_used == "stub-model"

    def test_sends_tokenized_prompt_to_llm_never_raw_pii(self):
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(VALID_RESPONSE)
        agent = PrimaryAgent(llm, self.vault)

        agent.run(tokenized)

        assert "123-45-6789" not in llm.last_prompt
        assert llm.call_count == 1

    def test_strips_markdown_code_fences_from_response(self):
        fenced = f"```json\n{VALID_RESPONSE}\n```"
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(fenced)
        agent = PrimaryAgent(llm, self.vault)

        result = agent.run(tokenized)

        assert result.confidence == 0.92

    def test_raises_on_invalid_json(self):
        tokenized = self._tokenized_doc()
        llm = StubLLMClient("this is not json")
        agent = PrimaryAgent(llm, self.vault)

        with pytest.raises(LLMOutputParseError):
            agent.run(tokenized)

    def test_raises_on_missing_required_field(self):
        incomplete = json.dumps({"extracted_fields": {}, "confidence": 0.5})
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(incomplete)
        agent = PrimaryAgent(llm, self.vault)

        with pytest.raises(LLMOutputParseError):
            agent.run(tokenized)

    def test_privacy_gate_aborts_call_if_leak_detected(self):
        """If the vault's leak assertion fires, the LLM must never be called."""
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(VALID_RESPONSE)
        agent = PrimaryAgent(llm, self.vault)

        # Simulate a leak by monkeypatching the vault's assertion to always fail
        def _always_fails(*args, **kwargs):
            raise VaultSecurityError("simulated leak")

        self.vault.assert_no_pii_leak = _always_fails

        with pytest.raises(VaultSecurityError):
            agent.run(tokenized)

        assert llm.call_count == 0

    def test_strips_unlabeled_markdown_code_fences(self):
        """Verifies code fence stripping works when the 'json' language specifier is omitted."""
        fenced = f"```\n{VALID_RESPONSE}\n```"
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(fenced)
        agent = PrimaryAgent(llm, self.vault)

        result = agent.run(tokenized)

        assert result.confidence == 0.92

    def test_raises_on_non_dictionary_json_response(self):
        """Verifies LLMOutputParseError is raised when LLM returns a valid JSON array instead of an object."""
        array_json = json.dumps(["field1", "field2"])
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(array_json)
        agent = PrimaryAgent(llm, self.vault)

        with pytest.raises(LLMOutputParseError):
            agent.run(tokenized)

    def test_raises_on_invalid_field_type_in_json(self):
        """Verifies LLMOutputParseError is raised when confidence is a string instead of a float."""
        invalid_type_json = json.dumps(
            {
                "extracted_fields": {},
                "inconsistencies": [],
                "missing_compliance_items": [],
                "confidence": "ninety-two-percent",  # Invalid type
                "reasoning": "Valid reasoning",
            }
        )
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(invalid_type_json)
        agent = PrimaryAgent(llm, self.vault)

        with pytest.raises(LLMOutputParseError):
            agent.run(tokenized)

    def test_default_model_name_assigned(self):
        """Verifies default model_name is set when no explicit model_name is provided to PrimaryAgent."""
        tokenized = self._tokenized_doc()
        llm = StubLLMClient(VALID_RESPONSE)
        agent = PrimaryAgent(llm, self.vault)

        result = agent.run(tokenized)

        assert result.model_used is not None
        assert isinstance(result.model_used, str)
