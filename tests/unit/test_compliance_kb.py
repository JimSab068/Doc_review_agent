import os
import pytest
from unittest.mock import patch, MagicMock
from src.compliance_kb import ComplianceKB, CompliancePassage
import numpy as np

# A quick stub class to simulate Chroma's embedding function behavior offline
# Update the stub class inside tests/unit/test_compliance_kb.py
class MockGeminiEmbeddingFunction:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input):
        """Called when embedding documents during data seeding."""
        return np.array([[0.1] * 768 for _ in input])

    def embed_query(self, input):
        """Called when embedding a search query text."""
        # Handles single strings or lists of strings safely
        if isinstance(input, str):
            return np.array([[0.1] * 768])
        return np.array([[0.1] * 768 for _ in input])

    def name(self) -> str:
        """Satisfies Chroma's collection configuration validations."""
        return "GoogleGeminiEmbeddingFunction"

@pytest.fixture(autouse=True)
def mock_ambient_env():
    """Ensures the environment check passes during unit testing without real keys."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-local-test-key"}):
        yield

@pytest.fixture
def mock_gemini_embedding():
    """Patches ChromaDB's remote embedding call hook to keep unit tests 100% offline."""
    with patch('chromadb.utils.embedding_functions.GoogleGeminiEmbeddingFunction', 
               side_effect=MockGeminiEmbeddingFunction) as mock_cls:
        yield mock_cls

import uuid

@pytest.fixture
def compliance_kb(mock_gemini_embedding):
    """Provides a clean, isolated in-memory knowledge base per test."""
    kb = ComplianceKB(collection_name=f"test_collection_{uuid.uuid4().hex}")
    yield kb
    try:
        kb.client.delete_collection(kb.collection.name)
    except Exception:
        pass

def test_seed_and_structural_integrity(compliance_kb):
    """Verifies that items are formatted and written into ChromaDB properly."""
    test_passage = CompliancePassage(
        id="rule_1",
        content="A creditor shall not discriminate on a prohibited basis.",
        citation="Reg B 1002.4(a)",
        metadata={"category": "compliance"}
    )
    
    compliance_kb.seed_initial_compliance_data([test_passage])
    
    # Check that it successfully landed in the underlying mock collection
    stored = compliance_kb.collection.get(ids=["rule_1"])
    assert stored["ids"][0] == "rule_1"
    assert stored["documents"][0] == test_passage.content

def test_query_routing_and_mapping(compliance_kb):
    """Verifies parsing logic returns formatted structural objects from raw DB payloads."""
    test_passage = CompliancePassage(
        id="rule_2",
        content="Inquiries regarding marital status are restricted.",
        citation="Reg B 1002.5(b)",
        metadata={"category": "restrictions"}
    )
    compliance_kb.seed_initial_compliance_data([test_passage])
    
    # Query using similar semantic text
    results = compliance_kb.query_relevant_policies("Checking marital status patterns", n_results=1)
    
    assert len(results) == 1
    assert results[0].id == "rule_2"
    assert results[0].citation == "Reg B 1002.5(b)"

def test_missing_api_key_raises_error():
    """Verifies that initializing ComplianceKB without GEMINI_API_KEY raises an error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises((ValueError, KeyError)):
            ComplianceKB(collection_name="no_key_test")


def test_query_empty_collection_returns_empty_list(compliance_kb):
    """Verifies that querying an unseeded collection returns an empty list without crashing."""
    results = compliance_kb.query_relevant_policies("Any policy query", n_results=3)
    assert results == []


def test_seed_and_query_multiple_passages_with_metadata(compliance_kb):
    """Verifies batch seeding, n_results slicing, and metadata preservation."""
    passages = [
        CompliancePassage(
            id="rule_a",
            content="Adverse action notices required within 30 days.",
            citation="Reg B 1002.9(a)",
            metadata={"type": "notice", "timeframe": "30d"},
        ),
        CompliancePassage(
            id="rule_b",
            content="Credit reports require permissible purpose.",
            citation="FCRA 604",
            metadata={"type": "credit", "timeframe": "immediate"},
        ),
    ]
    compliance_kb.seed_initial_compliance_data(passages)

    results = compliance_kb.query_relevant_policies("adverse action timing", n_results=2)
    assert len(results) == 2
    
    # Locate rule_a and verify metadata mapping
    rule_a = next(r for r in results if r.id == "rule_a")
    assert rule_a.citation == "Reg B 1002.9(a)"
    assert rule_a.metadata.get("type") == "notice"


def test_upsert_replaces_existing_passage(compliance_kb):
    """Verifies re-seeding the same ID updates existing content rather than throwing duplicate errors."""
    initial = CompliancePassage(
        id="rule_update",
        content="Original content string.",
        citation="Reg B 1002.1",
        metadata={},
    )
    updated = CompliancePassage(
        id="rule_update",
        content="Updated content string.",
        citation="Reg B 1002.1",
        metadata={},
    )

    compliance_kb.seed_initial_compliance_data([initial])
    compliance_kb.seed_initial_compliance_data([updated])

    results = compliance_kb.query_relevant_policies("content", n_results=1)
    assert results[0].content == "Updated content string."

