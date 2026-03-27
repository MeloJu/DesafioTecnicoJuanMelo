"""
EmbeddingService — indexação e retrieval semântico via ChromaDB + nomic-embed-text.

index(): converte List[Chunk] em embeddings e persiste no ChromaDB.
query(): embeds o texto de busca e retorna os chunks mais similares.

Dependências injetadas (ADR-006):
  chroma_client    → chromadb.PersistentClient em produção, Mock em testes
  embedding_client → OllamaEmbedder em produção, Mock em testes

Uma collection por empresa no ChromaDB (ADR-002):
  construtiva_engenharia, logitrans_global, vitalcare, ...
"""
from typing import List

from app.logging.logger import get_logger
from app.schemas.output import Chunk, ScoredChunk

log = get_logger()


class EmbeddingService:
    def __init__(self, chroma_client, embedding_client):
        self._chroma = chroma_client
        self._embedder = embedding_client

    def index(self, chunks: List[Chunk], collection: str) -> None:
        if not chunks:
            return

        col = self._chroma.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

        for chunk in chunks:
            embedding = self._embedder.embed(chunk.text)
            col.add(
                ids=[chunk.chunk_id],
                embeddings=[embedding],
                documents=[chunk.text],
                metadatas=[{
                    "empresa": chunk.empresa,
                    "setor": chunk.setor,
                    "source": chunk.source,
                }],
            )

        log.info(
            "chunks_indexed",
            collection=collection,
            total=len(chunks),
        )

    def query(self, text: str, collection: str, top_k: int = 5) -> List[ScoredChunk]:
        col = self._chroma.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

        count = col.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        embedding = self._embedder.embed(text)
        results = col.query(query_embeddings=[embedding], n_results=n_results)

        scored_chunks = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text_content, meta, distance in zip(ids, documents, metadatas, distances):
            chunk = Chunk(
                text=text_content,
                source=meta["source"],
                empresa=meta["empresa"],
                setor=meta["setor"],
                chunk_id=chunk_id,
            )
            scored_chunks.append(ScoredChunk(chunk=chunk, score=1.0 - distance))

        return scored_chunks
