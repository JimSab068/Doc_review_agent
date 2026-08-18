import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.schemas import (
    Document,
    DocumentType,
    RoutingDecision,
    DraftDecision,
)
from src.vault import VaultClient
from src.compliance_kb import ComplianceKB
from src.compliance_fixtures import REG_B_FCRA_FIXTURES
from src.pipeline import SecureAuditPipeline


class MockGeminiEmbeddingFunction:
    """Offline stub for ChromaDB embedding generation."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input):
        return np.array([[0.1] * 768 for _ in input])

    def embed_query(self, input):
        if isinstance(input, str):
            return  np.array([[0.1] * 768])
        return np.array([[0.1] * 768 for _ in input])

    def name(self):
        return "GoogleGeminiEmbeddingFunction"


@pytest.fixture(autouse=True)
def mock_ambient_env():
    with patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "mock-local-test-key"},
        clear=False,
    ):
        os.environ.pop("CHROMA_HOST", None)
        os.environ.pop("CHROMA_PORT", None)
        os.environ.pop("CHROMA_SSL", None)
        yield


@pytest.fixture(autouse=True)
def mock_gemini_embedding():
    with patch(
        "chromadb.utils.embedding_functions.GoogleGeminiEmbeddingFunction",
        side_effect=MockGeminiEmbeddingFunction,
    ):
        yield


@pytest.fixture
def seeded_kb():
    kb = ComplianceKB(collection_name="test_critic_kb")
    kb.seed_initial_compliance_data(REG_B_FCRA_FIXTURES)

    yield kb

    try:
        kb.client.delete_collection(kb.collection.name)
    except Exception:
        pass

def make_llm_response(response_text: str):
    resp = MagicMock()
    resp.text = response_text
    resp.content = response_text

    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = response_text
    resp.choices = [choice]

    resp.__str__.return_value = response_text
    return resp


def create_mock_llm(responses):
    mock_llm = MagicMock()

    wrapped = iter(
        [make_llm_response(r) if isinstance(r, str) else r for r in responses]
    )
    raw = iter(responses)

    def get_raw(*args, **kwargs):
        try:
            return next(raw)
        except StopIteration:
            return responses[-1]

    def get_wrapped(*args, **kwargs):
        try:
            return next(wrapped)
        except StopIteration:
            return make_llm_response(responses[-1])

    mock_llm.generate.side_effect = get_raw
    mock_llm.generate_content.side_effect = get_wrapped
    mock_llm.run.side_effect = get_wrapped
    mock_llm.invoke.side_effect = get_wrapped
    mock_llm.predict.side_effect = get_wrapped
    mock_llm.chat.side_effect = get_wrapped
    mock_llm.chat.completions.create.side_effect = get_wrapped
    mock_llm.beta.chat.completions.parse.side_effect = get_wrapped
    mock_llm.messages.create.side_effect = get_wrapped
    mock_llm.models.generate_content.side_effect = get_wrapped
    mock_llm.client.models.generate_content.side_effect = get_wrapped

    return mock_llm


async def run_pipeline_flow(
    pipeline,
    vault,
    seeded_kb,
    raw_content,
    doc_type,
    filename="local_loan.txt",
    threshold=0.85,
):
    doc = Document(
        doc_id="test_doc",
        doc_type=doc_type,
        source_filename=filename,
        pages=[{"page_number": 1, "text": raw_content}],
    )

    tokenized_doc = vault.tokenize_document(doc, spans=[])

    primary_decision = pipeline.primary_agent.run(tokenized_doc)

    if isinstance(primary_decision, str):
        primary_decision = DraftDecision.model_validate_json(primary_decision)
    elif not isinstance(primary_decision, DraftDecision):
        raw_text = (
            getattr(primary_decision, "text", None)
            or getattr(primary_decision, "content", None)
            or str(primary_decision)
        )

        if isinstance(raw_text, MagicMock):
            raw_text = str(raw_text)

        primary_decision = DraftDecision.model_validate_json(str(raw_text))

    passages = seeded_kb.query_relevant_policies(primary_decision.reasoning)

    critic_verdict = pipeline.critic_agent.evaluate(
        primary_decision,
        passages,
        tokenized_doc.token_map_id,
    )

    reasons = []

    if primary_decision.confidence < threshold:
        reasons.append(
            f"Primary confidence ({primary_decision.confidence}) below threshold ({threshold})"
        )

    if critic_verdict.verdict != "pass" or critic_verdict.escalate:
        reasons.append("Critic agent policy flagged or requested escalation")

    route = "human_queue" if reasons else "auto_resolve"

    routing = RoutingDecision(
        route=route,
        escalation_reason="; ".join(reasons),
        primary_confidence=primary_decision.confidence,
        critic_escalated=critic_verdict.escalate,
    )

    return primary_decision, critic_verdict, routing


@pytest.mark.asyncio
async def test_critic_gated_escalation_rules(seeded_kb):
    vault = VaultClient()

    primary_low_conf = """{
        "doc_id":"test_doc",
        "extracted_fields":{"SSN":"[[pii_ssn_1]]"},
        "inconsistencies":[],
        "missing_compliance_items":[],
        "confidence":0.80,
        "reasoning":"Marginal confidence."
    }"""

    critic_pass = """{
        "verdict":"pass",
        "cited_policy":["12 CFR § 1002.9"],
        "concerns":[],
        "escalate":false
    }"""

    pipeline = SecureAuditPipeline(
        create_mock_llm([primary_low_conf, critic_pass]),
        vault,
        seeded_kb,
    )

    _, _, routing = await run_pipeline_flow(
        pipeline,
        vault,
        seeded_kb,
        raw_content="Applicant SSN: 000-12-3456",
        doc_type=DocumentType.LOAN_APPLICATION,
    )

    assert routing.route == "human_queue"
    assert routing.primary_confidence == 0.80
    assert routing.critic_escalated is False
    assert "confidence" in routing.escalation_reason.lower()

    primary_high = """{
        "doc_id":"test_doc",
        "extracted_fields":{"SSN":"[[pii_ssn_1]]"},
        "inconsistencies":[],
        "missing_compliance_items":[],
        "confidence":0.95,
        "reasoning":"Strong parse."
    }"""

    critic_flag = """{
        "verdict":"flag",
        "cited_policy":["15 U.S.C. § 1681m"],
        "concerns":["Required adverse disclosures missing."],
        "escalate":false
    }"""

    pipeline = SecureAuditPipeline(
        create_mock_llm([primary_high, critic_flag]),
        vault,
        seeded_kb,
    )

    _, _, routing = await run_pipeline_flow(
        pipeline,
        vault,
        seeded_kb,
        raw_content="Applicant SSN: 000-12-3456",
        doc_type=DocumentType.LOAN_APPLICATION,
    )

    assert routing.route == "human_queue"
    assert routing.primary_confidence == 0.95
    assert routing.critic_escalated is False
    assert "policy" in routing.escalation_reason.lower()


@pytest.mark.asyncio
async def test_critic_auto_resolve_happy_path(seeded_kb):
    vault = VaultClient()

    primary = """{
        "doc_id":"test_doc",
        "extracted_fields":{},
        "inconsistencies":[],
        "missing_compliance_items":[],
        "confidence":0.99,
        "reasoning":"Clear document."
    }"""

    critic = """{
        "verdict":"pass",
        "cited_policy":["12 CFR §1002.9"],
        "concerns":[],
        "escalate":false
    }"""

    pipeline = SecureAuditPipeline(
        create_mock_llm([primary, critic]),
        vault,
        seeded_kb,
    )

    _, _, routing = await run_pipeline_flow(
        pipeline,
        vault,
        seeded_kb,
        raw_content="Applicant SSN: 000-12-3456",
        doc_type=DocumentType.LOAN_APPLICATION,
    )

    assert routing.route == "auto_resolve"
    assert routing.primary_confidence == 0.99
    assert routing.critic_escalated is False


@pytest.mark.asyncio
async def test_critic_fails_closed_on_invalid_json(seeded_kb):
    vault = VaultClient()

    primary = """{
        "doc_id":"test_doc",
        "extracted_fields":{},
        "inconsistencies":[],
        "missing_compliance_items":[],
        "confidence":0.95,
        "reasoning":"Valid primary parse."
    }"""

    pipeline = SecureAuditPipeline(
        create_mock_llm(
            [primary, "I am unable to evaluate this document in JSON format."]
        ),
        vault,
        seeded_kb,
    )

    _, critic, _ = await run_pipeline_flow(
        pipeline,
        vault,
        seeded_kb,
        raw_content="Clean text",
        doc_type=DocumentType.LOAN_APPLICATION,
    )

    assert critic.verdict == "flag"
    assert critic.escalate is True


@pytest.mark.asyncio
async def test_critic_explicit_escalation_flag_routes_to_human_queue(seeded_kb):
    vault = VaultClient()

    primary = """{
        "doc_id":"test_doc",
        "extracted_fields":{},
        "inconsistencies":[],
        "missing_compliance_items":[],
        "confidence":0.99,
        "reasoning":"Clear document."
    }"""

    critic = """{
        "verdict":"pass",
        "cited_policy":[],
        "concerns":["Edge case requires manual verification"],
        "escalate":true
    }"""

    pipeline = SecureAuditPipeline(
        create_mock_llm([primary, critic]),
        vault,
        seeded_kb,
    )

    _, _, routing = await run_pipeline_flow(
        pipeline,
        vault,
        seeded_kb,
        raw_content="Clean text",
        doc_type=DocumentType.LOAN_APPLICATION,
    )

    assert routing.route == "human_queue"
    assert routing.critic_escalated is True