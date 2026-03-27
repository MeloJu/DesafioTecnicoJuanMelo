"""
E2E Test 1: Worker with full PPE → status Conforme.

Pipeline completo com modelos reais:
  YOLO (detecção) + CLIP (atributos) + Ollama/nomic (embedding) + Ollama/Llama (reasoning)

Pré-requisitos:
  1. Ollama rodando com llama3.2 e nomic-embed-text
  2. tests/fixtures/images/worker_conforme.jpg — foto real de trabalhador com
     capacete, colete e botas de segurança visíveis

Para rodar:
    pytest tests/e2e/test_e2e_conforme.py -m e2e -s
"""
import pytest

from app.pipeline.factory import create_pipeline
from app.schemas.output import PipelineResponse

from tests.e2e.conftest import (
    requires_ollama,
    skip_if_image_missing,
    _image_path,
)

IMAGE_FILE = "worker_conforme.jpg"


@requires_ollama
@skip_if_image_missing(IMAGE_FILE)
@pytest.mark.e2e
def test_e2e_worker_conforme(indexed_fixture_documents):
    """
    Pessoa com todos os EPIs obrigatórios → pipeline retorna Conforme.

    O teste verifica:
    - Pipeline não lança exceção
    - Pelo menos uma pessoa é detectada
    - Cada PersonResult tem status válido e justificativa não vazia
    - Status esperado: "Conforme"
    """
    image_path = str(_image_path(IMAGE_FILE))
    pipeline = create_pipeline(chroma_path=indexed_fixture_documents)

    response = pipeline.run(image_path, empresa="TestEmpresa", setor="obras")

    assert isinstance(response, PipelineResponse), (
        f"Pipeline deve retornar PipelineResponse, recebeu {type(response)}"
    )
    assert len(response.results) > 0, (
        "Nenhuma pessoa detectada — verifique se a imagem contém um trabalhador visível"
    )

    for result in response.results:
        assert result.status in {"Conforme", "Não conforme", "Indeterminado"}, (
            f"Status inválido: {result.status!r}"
        )
        assert result.justificativa.strip(), (
            f"justificativa vazia para pessoa {result.pessoa_id}"
        )
        assert result.bbox.x2 > result.bbox.x1
        assert result.bbox.y2 > result.bbox.y1

    # Para trabalhador com todos EPIs, esperamos Conforme
    statuses = {r.status for r in response.results}
    assert "Conforme" in statuses, (
        f"Esperado ao menos um 'Conforme', obtidos: {statuses}\n"
        f"Justificativas: {[r.justificativa for r in response.results]}"
    )
