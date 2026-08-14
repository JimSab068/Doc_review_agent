import os
from unittest.mock import patch

import pytest

from src.secret_redaction import redact_secrets, safe_exception_message


class TestRedactSecrets:
    def test_redacts_gemini_api_key_value(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-super-secret-value-123"}):
            text = "request failed, key used was sk-super-secret-value-123 in header"
            result = redact_secrets(text)

            assert "sk-super-secret-value-123" not in result
            assert "[REDACTED:GEMINI_API_KEY]" in result

    def test_redacts_google_api_key_value(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "another-secret-value-456"}):
            text = "auth header: Bearer another-secret-value-456"
            result = redact_secrets(text)

            assert "another-secret-value-456" not in result

    def test_leaves_clean_text_untouched(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-super-secret-value-123"}):
            text = "model output was not valid JSON"
            assert redact_secrets(text) == text

    def test_does_not_redact_short_values_that_could_mangle_text(self):
        # A very short "key" shouldn't cause redact_secrets to eat
        # unrelated substrings that happen to match it.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "ab"}):
            text = "the word ab shows up here normally"
            assert redact_secrets(text) == text

    def test_handles_missing_env_vars_gracefully(self):
        with patch.dict(os.environ, {}, clear=True):
            text = "some error text"
            assert redact_secrets(text) == text

    def test_safe_exception_message_redacts(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-super-secret-value-123"}):
            exc = ValueError("bad request with key sk-super-secret-value-123 attached")
            result = safe_exception_message(exc)

            assert "sk-super-secret-value-123" not in result


class TestGeminiClientNeverExposesKey:
    def test_generate_wraps_sdk_error_and_redacts_key(self):
        """Simulates the failure mode this whole fix targets: the SDK
        raises an exception whose text happens to contain the raw key
        (as some HTTP-layer failures do), and confirms GeminiClient.generate
        never lets that reach the caller unredacted."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-super-secret-value-123"}):
            from src.primary_agent import GeminiClient, LLMAPIError

            with patch("google.genai.Client") as mock_client_cls:
                mock_instance = mock_client_cls.return_value
                mock_instance.models.generate_content.side_effect = RuntimeError(
                    "connection failed, sent header Authorization: sk-super-secret-value-123"
                )

                client = GeminiClient(model_name="gemini-3.1-flash-lite")

                with pytest.raises(LLMAPIError) as exc_info:
                    client.generate("some prompt")

                assert "sk-super-secret-value-123" not in str(exc_info.value)

    def test_constructor_never_accepts_api_key_argument(self):
        """Regression guard for the bug in the original manual test script:
        GeminiClient must never accept a credential as a parameter, since
        anything passed as an argument can end up in a repr, a log line,
        or a traceback further up the call stack."""
        from src.primary_agent import GeminiClient

        with pytest.raises(TypeError):
            GeminiClient(api_key="should-not-be-accepted")