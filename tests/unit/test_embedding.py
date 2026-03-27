"""
Unit tests for the EmbeddingService.

What we test (ChromaDB and Ollama always mocked):
- index(): gera embedding para cada chunk, persiste no ChromaDB
- index(): chunks vazios → sem chamadas a embed ou add
- index(): metadados corretos passados ao ChromaDB
- query(): embeds o texto de busca, consulta ChromaDB, retorna List[ScoredChunk]
- query(): top_k respeitado na chamada ao ChromaDB
- query(): score calculado a partir da distância (1 - distance)
- query(): coleção vazia → lista vazia sem erro
- Coleção correta usada em index e query
"""
import pytest
from unittest.mock import Mock, MagicMock, call

from app.rag.embedding_service import EmbeddingService
from app.schemas.output import Chunk, ScoredChunk

COLLECTION = "construtiva_engenharia"
FAKE_EMBEDDING = [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str = "id-1", text: str = "Capacete obrigatório em obras.") -> Chunk:
    return Chunk(
        text=text,
        source="manual.pdf",
        empresa="Construtiva",
        setor="obras",
        chunk_id=chunk_id,
    )


def _make_chroma_results(ids, documents, metadatas, distances):
    """Build the dict structure ChromaDB returns from collection.query()."""
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _make_service(chroma_client=None, embedding_client=None):
    chroma_client = chroma_client or Mock()
    embedding_client = embedding_client or Mock()
    embedding_client.embed.return_value = FAKE_EMBEDDING
    return EmbeddingService(
        chroma_client=chroma_client,
        embedding_client=embedding_client,
    )


# ---------------------------------------------------------------------------
# index()
# ---------------------------------------------------------------------------

class TestIndex:
    def test_calls_embed_for_each_chunk(self):
        mock_embedder = Mock()
        mock_embedder.embed.return_value = FAKE_EMBEDDING
        mock_col = Mock()
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = EmbeddingService(mock_chroma, mock_embedder)
        chunks = [_make_chunk("id-1"), _make_chunk("id-2")]
        service.index(chunks, COLLECTION)

        assert mock_embedder.embed.call_count == 2

    def test_calls_chroma_add_for_each_chunk(self):
        mock_embedder = Mock()
        mock_embedder.embed.return_value = FAKE_EMBEDDING
        mock_col = Mock()
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = EmbeddingService(mock_chroma, mock_embedder)
        chunks = [_make_chunk("id-1"), _make_chunk("id-2")]
        service.index(chunks, COLLECTION)

        assert mock_col.add.call_count == 2

    def test_uses_correct_collection_name(self):
        mock_col = Mock()
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col
        service = _make_service(chroma_client=mock_chroma)

        service.index([_make_chunk()], COLLECTION)

        mock_chroma.get_or_create_collection.assert_called_once()
        call_args = mock_chroma.get_or_create_collection.call_args
        assert COLLECTION in call_args[0] or call_args[1].get("name") == COLLECTION

    def test_chunk_id_used_as_chroma_id(self):
        mock_embedder = Mock()
        mock_embedder.embed.return_value = FAKE_EMBEDDING
        mock_col = Mock()
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = EmbeddingService(mock_chroma, mock_embedder)
        service.index([_make_chunk("my-uuid-123")], COLLECTION)

        call_kwargs = mock_col.add.call_args[1]
        assert "my-uuid-123" in call_kwargs["ids"]

    def test_empty_chunks_no_calls(self):
        mock_embedder = Mock()
        mock_chroma = Mock()
        service = EmbeddingService(mock_chroma, mock_embedder)

        service.index([], COLLECTION)

        mock_embedder.embed.assert_not_called()
        mock_chroma.get_or_create_collection.assert_not_called()

    def test_metadata_passed_to_chroma(self):
        mock_embedder = Mock()
        mock_embedder.embed.return_value = FAKE_EMBEDDING
        mock_col = Mock()
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        chunk = _make_chunk()
        service = EmbeddingService(mock_chroma, mock_embedder)
        service.index([chunk], COLLECTION)

        call_kwargs = mock_col.add.call_args[1]
        meta = call_kwargs["metadatas"][0]
        assert meta["empresa"] == chunk.empresa
        assert meta["setor"] == chunk.setor
        assert meta["source"] == chunk.source


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestQuery:
    def test_returns_list_of_scored_chunks(self):
        mock_col = Mock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = _make_chroma_results(
            ids=["id-1"],
            documents=["Capacete obrigatório em obras."],
            metadatas=[{"empresa": "Construtiva", "setor": "obras", "source": "manual.pdf"}],
            distances=[0.1],
        )
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = _make_service(chroma_client=mock_chroma)
        results = service.query("capacete obras", COLLECTION, top_k=3)

        assert isinstance(results, list)
        assert all(isinstance(r, ScoredChunk) for r in results)

    def test_score_is_one_minus_distance(self):
        mock_col = Mock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = _make_chroma_results(
            ids=["id-1"],
            documents=["Capacete obrigatório em obras."],
            metadatas=[{"empresa": "Construtiva", "setor": "obras", "source": "manual.pdf"}],
            distances=[0.2],
        )
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = _make_service(chroma_client=mock_chroma)
        results = service.query("capacete", COLLECTION, top_k=1)

        assert abs(results[0].score - 0.8) < 1e-9

    def test_top_k_passed_to_chroma(self):
        mock_col = Mock()
        mock_col.count.return_value = 10
        mock_col.query.return_value = _make_chroma_results([], [], [], [])
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = _make_service(chroma_client=mock_chroma)
        service.query("capacete", COLLECTION, top_k=7)

        call_kwargs = mock_col.query.call_args[1]
        assert call_kwargs["n_results"] == 7

    def test_empty_collection_returns_empty_list(self):
        mock_col = Mock()
        mock_col.count.return_value = 0
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = _make_service(chroma_client=mock_chroma)
        results = service.query("capacete", COLLECTION, top_k=5)

        assert results == []
        mock_col.query.assert_not_called()

    def test_chunk_text_and_metadata_preserved(self):
        mock_col = Mock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = _make_chroma_results(
            ids=["id-abc"],
            documents=["Capacete obrigatório em obras."],
            metadatas=[{"empresa": "Construtiva", "setor": "obras", "source": "manual.pdf"}],
            distances=[0.05],
        )
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = _make_service(chroma_client=mock_chroma)
        results = service.query("capacete", COLLECTION, top_k=1)

        chunk = results[0].chunk
        assert chunk.chunk_id == "id-abc"
        assert chunk.text == "Capacete obrigatório em obras."
        assert chunk.empresa == "Construtiva"
        assert chunk.setor == "obras"

    def test_top_k_capped_at_collection_size(self):
        mock_col = Mock()
        mock_col.count.return_value = 2  # apenas 2 docs na coleção
        mock_col.query.return_value = _make_chroma_results([], [], [], [])
        mock_chroma = Mock()
        mock_chroma.get_or_create_collection.return_value = mock_col

        service = _make_service(chroma_client=mock_chroma)
        service.query("capacete", COLLECTION, top_k=10)

        call_kwargs = mock_col.query.call_args[1]
        assert call_kwargs["n_results"] == 2
