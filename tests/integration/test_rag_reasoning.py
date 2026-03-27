"""
Integration test: RagService + ReasoningService.

Módulos reais integrados. Dependências externas mockadas:
  - ChromaDB (in-memory mock)
  - OllamaEmbedder (retorna vetores fixos)
  - OllamaLLM (retorna JSON fixo)

Verifica que a fachada RagService recupera regras e que o ReasoningService
as analisa corretamente, produzindo PersonResult com status e justificativa.
"""
import json
from unittest.mock import MagicMock, Mock

import pytest

from app.rag.embedding_service import EmbeddingService
from app.rag.service import RagService
from app.reasoning.service import ReasoningService
from app.schemas.epi_config import EPIAttribute
from app.schemas.output import (
    BoundingBox,
    Chunk,
    PersonDetection,
)

CORRELATION_ID = "integration-rag-reasoning-id"

_FIXED_EMBEDDING = [0.1] * 128

_DEFAULT_EPI = [
    EPIAttribute("helmet", "capacete", "wearing hard hat", "not wearing hard hat"),
    EPIAttribute("vest", "colete refletivo", "wearing vest", "not wearing vest"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_person(helmet: bool = False, vest: bool = True) -> PersonDetection:
    return PersonDetection(
        pessoa_id=1,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=200),
        attributes={"helmet": helmet, "vest": vest},
    )


def _make_chunk(
    text: str = "Capacete obrigatório em todas as áreas de obras.",
    source: str = "normas_obras.pdf",
    empresa: str = "Construtiva",
    setor: str = "obras",
    chunk_id: str = "chunk-001",
) -> Chunk:
    return Chunk(
        text=text,
        source=source,
        empresa=empresa,
        setor=setor,
        chunk_id=chunk_id,
    )


def _make_chroma_mock(chunks: list[Chunk]) -> MagicMock:
    """Returns a ChromaDB client mock pre-loaded with the given chunks."""
    col = MagicMock()
    col.count.return_value = len(chunks)

    if chunks:
        col.query.return_value = {
            "ids": [[c.chunk_id for c in chunks]],
            "documents": [[c.text for c in chunks]],
            "metadatas": [[{"source": c.source, "empresa": c.empresa, "setor": c.setor} for c in chunks]],
            "distances": [[0.1] * len(chunks)],
        }
    else:
        col.count.return_value = 0

    chroma = MagicMock()
    chroma.get_or_create_collection.return_value = col
    return chroma


def _make_embedder_mock() -> Mock:
    embedder = Mock()
    embedder.embed.return_value = _FIXED_EMBEDDING
    return embedder


def _build_rag_service(chunks: list[Chunk]) -> RagService:
    embedding_service = EmbeddingService(
        chroma_client=_make_chroma_mock(chunks),
        embedding_client=_make_embedder_mock(),
    )
    return RagService(embedding_service=embedding_service)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRagReasoningIntegration:

    def test_conforme_when_rules_retrieved_and_llm_says_conforme(self):
        """Fluxo completo: regras recuperadas → LLM retorna Conforme → PersonResult Conforme."""
        chunk = _make_chunk("Capacete obrigatório em todas as áreas de obras.")
        rag = _build_rag_service([chunk])

        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "status": "Conforme",
            "justificativa": "Capacete presente conforme norma obras.",
        })
        reasoning = ReasoningService(llm_client=mock_llm)

        person = _make_person(helmet=True, vest=True)
        rules = rag.retrieve("Construtiva", "obras", "capacete colete", CORRELATION_ID)

        assert len(rules) == 1
        assert "Capacete" in rules[0].rule
        assert rules[0].source == "normas_obras.pdf"

        result = reasoning.analyze(person, rules, CORRELATION_ID, _DEFAULT_EPI)

        assert result.status == "Conforme"
        assert result.pessoa_id == 1
        assert "Capacete" in result.justificativa

    def test_nao_conforme_when_llm_says_nao_conforme(self):
        """LLM retorna Não conforme → PersonResult reflete isso."""
        chunk = _make_chunk("Capacete obrigatório em todas as áreas de obras.")
        rag = _build_rag_service([chunk])

        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "status": "Não conforme",
            "justificativa": "Capacete ausente — viola norma obrigatória.",
        })
        reasoning = ReasoningService(llm_client=mock_llm)

        person = _make_person(helmet=False, vest=True)
        rules = rag.retrieve("Construtiva", "obras", "capacete", CORRELATION_ID)
        result = reasoning.analyze(person, rules, CORRELATION_ID, _DEFAULT_EPI)

        assert result.status == "Não conforme"
        assert "ausente" in result.justificativa.lower()

    def test_indeterminado_when_no_rules_in_collection(self):
        """Coleção vazia → RagService retorna [] → ReasoningService retorna Indeterminado."""
        rag = _build_rag_service([])  # empty collection
        mock_llm = Mock()
        reasoning = ReasoningService(llm_client=mock_llm)

        person = _make_person()
        rules = rag.retrieve("Construtiva", "obras", "capacete", CORRELATION_ID)

        assert rules == []

        result = reasoning.analyze(person, rules, CORRELATION_ID, _DEFAULT_EPI)

        assert result.status == "Indeterminado"
        mock_llm.generate.assert_not_called()

    def test_multiple_rules_all_passed_to_reasoning(self):
        """Múltiplas regras são recuperadas e passadas ao ReasoningService."""
        chunks = [
            _make_chunk(
                "Capacete obrigatório em todas as áreas de obras.",
                chunk_id="chunk-001",
            ),
            _make_chunk(
                "Colete reflexivo obrigatório em todas as obras noturnas.",
                chunk_id="chunk-002",
            ),
        ]
        chroma_col = MagicMock()
        chroma_col.count.return_value = len(chunks)
        chroma_col.query.return_value = {
            "ids": [[c.chunk_id for c in chunks]],
            "documents": [[c.text for c in chunks]],
            "metadatas": [[{"source": c.source, "empresa": c.empresa, "setor": c.setor} for c in chunks]],
            "distances": [[0.05, 0.10]],
        }
        chroma = MagicMock()
        chroma.get_or_create_collection.return_value = chroma_col

        embedding_service = EmbeddingService(
            chroma_client=chroma,
            embedding_client=_make_embedder_mock(),
        )
        rag = RagService(embedding_service=embedding_service)

        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "status": "Conforme",
            "justificativa": "Capacete e colete presentes conforme normas.",
        })
        reasoning = ReasoningService(llm_client=mock_llm)

        person = _make_person(helmet=True, vest=True)
        rules = rag.retrieve("Construtiva", "obras", "capacete colete", CORRELATION_ID)

        assert len(rules) == 2

        result = reasoning.analyze(person, rules, CORRELATION_ID, _DEFAULT_EPI)

        assert result.status == "Conforme"
        # LLM recebeu prompt com ambas as regras
        prompt_used = mock_llm.generate.call_args[0][0]
        assert "Capacete" in prompt_used
        assert "Colete" in prompt_used

    def test_indeterminado_when_llm_raises(self):
        """LLM lança exceção → ReasoningService retorna Indeterminado sem propagar."""
        chunk = _make_chunk()
        rag = _build_rag_service([chunk])

        mock_llm = Mock()
        mock_llm.generate.side_effect = ConnectionError("Ollama offline")
        reasoning = ReasoningService(llm_client=mock_llm)

        person = _make_person()
        rules = rag.retrieve("Construtiva", "obras", "capacete", CORRELATION_ID)
        result = reasoning.analyze(person, rules, CORRELATION_ID, _DEFAULT_EPI)

        assert result.status == "Indeterminado"
        assert "Ollama offline" in result.justificativa

    def test_collection_name_normalised_from_empresa(self):
        """Empresa com espaços/maiúsculas → collection name normalizado corretamente."""
        chunk = _make_chunk(empresa="Construtiva Engenharia")
        chroma = MagicMock()
        col = MagicMock()
        col.count.return_value = 1
        col.query.return_value = {
            "ids": [[chunk.chunk_id]],
            "documents": [[chunk.text]],
            "metadatas": [[{"source": chunk.source, "empresa": chunk.empresa, "setor": chunk.setor}]],
            "distances": [[0.1]],
        }
        chroma.get_or_create_collection.return_value = col

        embedding_service = EmbeddingService(
            chroma_client=chroma,
            embedding_client=_make_embedder_mock(),
        )
        rag = RagService(embedding_service=embedding_service)
        rag.retrieve("Construtiva Engenharia", "obras", "capacete", CORRELATION_ID)

        call_kwargs = chroma.get_or_create_collection.call_args
        collection_name = call_kwargs[1]["name"] if call_kwargs[1] else call_kwargs[0][0]
        assert collection_name == "construtiva_engenharia"
