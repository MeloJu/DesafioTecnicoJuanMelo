"""
Factory — monta o pipeline completo com todas as dependências reais (ADR-006).

Único ponto do sistema onde as implementações concretas são instanciadas.
Em testes, o pipeline é montado diretamente com mocks no __init__.

Uso:
    from app.pipeline.factory import create_pipeline
    pipeline = create_pipeline()
    response = pipeline.run("imagem.jpg", empresa="Construtiva", setor="obras")
"""
from typing import List, Optional

import chromadb

from app.pipeline.orchestrator import Pipeline
from app.rag.embedding_service import EmbeddingService
from app.rag.ollama_embedder import OllamaEmbedder
from app.rag.service import RagService
from app.reasoning.ollama_llm import OllamaLLM
from app.reasoning.service import ReasoningService
from app.schemas.epi_config import DEFAULT_EPI_ATTRIBUTES, EPIAttribute
from app.vision.clip_client import CLIPClient
from app.vision.detector import PersonDetector
from app.vision.extractor import AttributeExtractor
from app.vision.service import VisionService


def create_pipeline(  # pragma: no cover
    yolo_model: str = "yolov8n.pt",
    llm_model: str = "llama3.2",
    embed_model: str = "nomic-embed-text",
    chroma_path: str = "./chroma_db",
    epi_attributes: Optional[List[EPIAttribute]] = None,
    clip_model_path: Optional[str] = None,
) -> Pipeline:
    """Instancia e conecta todos os módulos do pipeline.

    clip_model_path: caminho para um modelo CLIP fine-tunado
        (salvo via CLIPModel.save_pretrained). Se None, usa o modelo base OpenAI.
        Exemplo: create_pipeline(clip_model_path="models/clip_ppe")
    """
    if epi_attributes is None:
        epi_attributes = DEFAULT_EPI_ATTRIBUTES

    # Vision
    detector = PersonDetector(model_path=yolo_model)
    extractor = AttributeExtractor(clip_client=CLIPClient(model_path=clip_model_path), epi_attributes=epi_attributes)
    vision_service = VisionService(detector=detector, extractor=extractor)

    # RAG
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    embedding_service = EmbeddingService(
        chroma_client=chroma_client,
        embedding_client=OllamaEmbedder(model=embed_model),
    )
    rag_service = RagService(embedding_service=embedding_service)

    # Reasoning
    reasoning_service = ReasoningService(llm_client=OllamaLLM(model=llm_model))

    return Pipeline(
        vision_service=vision_service,
        rag_service=rag_service,
        reasoning_service=reasoning_service,
        epi_attributes=epi_attributes,
    )
