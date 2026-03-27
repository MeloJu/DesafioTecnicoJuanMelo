"""
Unit tests for RagService.

What we test (EmbeddingService always mocked):
- retrieve() retorna List[Rule] construída a partir de ScoredChunks
- Coleção vazia → [] + log warning
- Nome da empresa normalizado para collection_name
- Regras com source correto
- retrieve() com múltiplas chunks → múltiplas Rule
"""
import pytest
from unittest.mock import Mock, patch

from app.rag.service import RagService, _collection_name
from app.schemas.output import Chunk, Rule, ScoredChunk

CORRELATION_ID = "test-rag-service-id"


def _make_scored_chunk(
    text: str = "Capacete obrigatório em todas as áreas de obras.",
    source: str = "normas.pdf",
    empresa: str = "Construtiva",
    setor: str = "obras",
    score: float = 0.9,
    chunk_id: str = "c1",
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            text=text,
            source=source,
            empresa=empresa,
            setor=setor,
            chunk_id=chunk_id,
        ),
        score=score,
    )


# ---------------------------------------------------------------------------
# _collection_name helper
# ---------------------------------------------------------------------------


class TestCollectionName:
    def test_lowercase_simple(self):
        assert _collection_name("Construtiva") == "construtiva"

    def test_spaces_replaced_by_underscore(self):
        assert _collection_name("Construtiva Engenharia") == "construtiva_engenharia"

    def test_special_chars_replaced(self):
        assert _collection_name("VitalCare S/A") == "vitalcare_s_a"

    def test_leading_trailing_underscores_stripped(self):
        assert _collection_name("_empresa_") == "empresa"

    def test_multiple_spaces_single_underscore(self):
        assert _collection_name("Log  Trans") == "log_trans"


# ---------------------------------------------------------------------------
# RagService.retrieve
# ---------------------------------------------------------------------------


class TestRagServiceRetrieve:
    def _make_service(self, scored_chunks: list) -> RagService:
        mock_embedder = Mock()
        mock_embedder.query.return_value = scored_chunks
        return RagService(embedding_service=mock_embedder)

    def test_returns_rules_from_scored_chunks(self):
        sc = _make_scored_chunk(text="Capacete obrigatório.", source="normas.pdf")
        service = self._make_service([sc])
        rules = service.retrieve("Construtiva", "obras", "capacete", CORRELATION_ID)
        assert len(rules) == 1
        assert isinstance(rules[0], Rule)
        assert rules[0].rule == "Capacete obrigatório."
        assert rules[0].source == "normas.pdf"

    def test_empty_collection_returns_empty_list(self):
        service = self._make_service([])
        rules = service.retrieve("Construtiva", "obras", "capacete", CORRELATION_ID)
        assert rules == []

    def test_multiple_chunks_become_multiple_rules(self):
        chunks = [
            _make_scored_chunk(text="Regra 1.", chunk_id="c1"),
            _make_scored_chunk(text="Regra 2.", chunk_id="c2"),
            _make_scored_chunk(text="Regra 3.", chunk_id="c3"),
        ]
        service = self._make_service(chunks)
        rules = service.retrieve("Construtiva", "obras", "epi", CORRELATION_ID)
        assert len(rules) == 3
        assert [r.rule for r in rules] == ["Regra 1.", "Regra 2.", "Regra 3."]

    def test_collection_name_passed_to_embedder(self):
        mock_embedder = Mock()
        mock_embedder.query.return_value = []
        service = RagService(embedding_service=mock_embedder)
        service.retrieve("Construtiva Engenharia", "obras", "capacete", CORRELATION_ID)
        call_args = mock_embedder.query.call_args
        assert call_args[0][1] == "construtiva_engenharia"

    def test_query_text_passed_to_embedder(self):
        mock_embedder = Mock()
        mock_embedder.query.return_value = []
        service = RagService(embedding_service=mock_embedder)
        service.retrieve("Construtiva", "obras", "capacete colete", CORRELATION_ID)
        call_args = mock_embedder.query.call_args
        assert call_args[0][0] == "capacete colete"

    def test_top_k_passed_to_embedder(self):
        mock_embedder = Mock()
        mock_embedder.query.return_value = []
        service = RagService(embedding_service=mock_embedder)
        service.retrieve("Construtiva", "obras", "capacete", CORRELATION_ID)
        call_kwargs = mock_embedder.query.call_args[1]
        assert call_kwargs.get("top_k") == 5
