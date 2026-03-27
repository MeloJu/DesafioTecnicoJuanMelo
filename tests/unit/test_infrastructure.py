"""
Unit tests for thin infrastructure wrappers.

Covers: OllamaEmbedder, OllamaLLM, CLIPClient.

Real external calls (ollama SDK, HuggingFace) are always mocked.
These tests verify that the wrappers correctly delegate to their
underlying clients and transform the response.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------


class TestOllamaEmbedder:
    def test_embed_returns_embedding_from_response(self):
        from app.rag.ollama_embedder import OllamaEmbedder

        fake_embedding = [0.1, 0.2, 0.3]
        with patch("app.rag.ollama_embedder.ollama") as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": fake_embedding}
            embedder = OllamaEmbedder(model="nomic-embed-text")
            result = embedder.embed("texto de teste")

        assert result == fake_embedding

    def test_embed_passes_model_and_text(self):
        from app.rag.ollama_embedder import OllamaEmbedder

        with patch("app.rag.ollama_embedder.ollama") as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": [0.0]}
            embedder = OllamaEmbedder(model="nomic-embed-text")
            embedder.embed("meu texto")

        mock_ollama.embeddings.assert_called_once_with(
            model="nomic-embed-text", prompt="meu texto"
        )

    def test_default_model_is_nomic(self):
        from app.rag.ollama_embedder import OllamaEmbedder

        with patch("app.rag.ollama_embedder.ollama") as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": []}
            embedder = OllamaEmbedder()
            embedder.embed("x")

        call_kwargs = mock_ollama.embeddings.call_args[1]
        assert call_kwargs["model"] == "nomic-embed-text"


# ---------------------------------------------------------------------------
# OllamaLLM
# ---------------------------------------------------------------------------


class TestOllamaLLM:
    def test_generate_returns_response_field(self):
        from app.reasoning.ollama_llm import OllamaLLM

        with patch("app.reasoning.ollama_llm.ollama") as mock_ollama:
            mock_ollama.generate.return_value = {"response": '{"status": "Conforme", "justificativa": "ok"}'}
            llm = OllamaLLM(model="llama3.2")
            result = llm.generate("meu prompt")

        assert result == '{"status": "Conforme", "justificativa": "ok"}'

    def test_generate_passes_model_and_prompt(self):
        from app.reasoning.ollama_llm import OllamaLLM

        with patch("app.reasoning.ollama_llm.ollama") as mock_ollama:
            mock_ollama.generate.return_value = {"response": "ok"}
            llm = OllamaLLM(model="llama3.2")
            llm.generate("analise esta pessoa")

        mock_ollama.generate.assert_called_once_with(
            model="llama3.2", prompt="analise esta pessoa"
        )

    def test_default_model_is_llama(self):
        from app.reasoning.ollama_llm import OllamaLLM

        with patch("app.reasoning.ollama_llm.ollama") as mock_ollama:
            mock_ollama.generate.return_value = {"response": "ok"}
            llm = OllamaLLM()
            llm.generate("x")

        call_kwargs = mock_ollama.generate.call_args[1]
        assert call_kwargs["model"] == "llama3.2"

    def test_correlation_id_accepted_but_not_forwarded(self):
        """correlation_id é aceito pela assinatura mas não enviado ao Ollama."""
        from app.reasoning.ollama_llm import OllamaLLM

        with patch("app.reasoning.ollama_llm.ollama") as mock_ollama:
            mock_ollama.generate.return_value = {"response": "ok"}
            llm = OllamaLLM()
            llm.generate("prompt", correlation_id="abc-123")

        # Ollama não recebe correlation_id
        call_kwargs = mock_ollama.generate.call_args[1]
        assert "correlation_id" not in call_kwargs


# ---------------------------------------------------------------------------
# CLIPClient
# ---------------------------------------------------------------------------


class TestCLIPClient:
    def _make_mock_outputs(self, positive_prob: float):
        """Build mock CLIP model outputs with given positive class probability."""
        import torch

        logits = torch.tensor([[positive_prob, 1.0 - positive_prob]])
        outputs = Mock()
        outputs.logits_per_image = logits
        return outputs

    @patch("app.vision.clip_client.CLIPProcessor")
    @patch("app.vision.clip_client.CLIPModel")
    def test_classify_returns_positive_score(self, mock_model_cls, mock_processor_cls):
        from app.vision.clip_client import CLIPClient

        import torch

        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_processor = Mock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"input_ids": Mock(), "pixel_values": Mock()}

        # Simulate softmax: [0.8, 0.2]
        logits = torch.tensor([[2.0, 0.5]])
        mock_outputs = Mock()
        mock_outputs.logits_per_image = logits
        mock_model.return_value = mock_outputs

        client = CLIPClient()
        score = client.classify(
            Image.new("RGB", (100, 100)),
            positive_text="a person wearing a hard hat",
            negative_text="a person not wearing a hard hat",
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch("app.vision.clip_client.CLIPProcessor")
    @patch("app.vision.clip_client.CLIPModel")
    def test_classify_passes_texts_to_processor(self, mock_model_cls, mock_processor_cls):
        from app.vision.clip_client import CLIPClient

        import torch

        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_processor = Mock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {}

        logits = torch.tensor([[1.0, 0.0]])
        mock_out = Mock()
        mock_out.logits_per_image = logits
        mock_model.return_value = mock_out

        client = CLIPClient()
        client.classify(
            Image.new("RGB", (100, 100)),
            positive_text="wearing hard hat",
            negative_text="not wearing hard hat",
        )

        call_kwargs = mock_processor.call_args[1]
        assert "wearing hard hat" in call_kwargs["text"]
        assert "not wearing hard hat" in call_kwargs["text"]

    @patch("app.vision.clip_client.CLIPProcessor")
    @patch("app.vision.clip_client.CLIPModel")
    def test_classify_with_custom_texts(self, mock_model_cls, mock_processor_cls):
        from app.vision.clip_client import CLIPClient

        import torch

        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_processor = Mock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {}

        logits = torch.tensor([[0.7, 0.3]])
        mock_out = Mock()
        mock_out.logits_per_image = logits
        mock_model.return_value = mock_out

        client = CLIPClient()
        image = Image.new("RGB", (100, 100))
        # Any EPI can now be classified by passing its texts directly
        score = client.classify(image, "wearing safety glasses", "not wearing safety glasses")
        assert isinstance(score, float)
