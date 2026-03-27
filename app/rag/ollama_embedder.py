"""
OllamaEmbedder — wrapper real para nomic-embed-text via Ollama.

Injetado no EmbeddingService via DI — nunca instanciado em testes unitários.
Requer Ollama rodando localmente com nomic-embed-text disponível.
"""
from typing import List

import ollama


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text"):
        self._model = model

    def embed(self, text: str) -> List[float]:
        response = ollama.embeddings(model=self._model, prompt=text)
        return response["embedding"]
