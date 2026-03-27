"""
E2E Test 2: Worker without PPE → status Não conforme.

Pipeline completo com modelos reais:
  YOLO (detecção) + CLIP (atributos) + Ollama/nomic (embedding) + Ollama/Llama (reasoning)

Pré-requisitos:
  1. Ollama rodando com llama3.2 e nomic-embed-text
  2. tests/fixtures/images/worker_nao_conforme.jpg — foto real de trabalhador
     SEM capacete (EPI ausente visível)

Para rodar:
    pytest tests/e2e/test_e2e_nao_conforme.py -m e2e -s
"""
import pytest

from app.pipeline.factory import create_pipeline
from app.schemas.output import PipelineResponse

from tests.e2e.conftest import (
    requires_ollama,
    skip_if_image_missing,
    _image_path,
)

IMAGE_FILE = "worker_nao_conforme.jpg"


@requires_ollama
@skip_if_image_missing(IMAGE_FILE)
@pytest.mark.e2e
def test_e2e_worker_nao_conforme(indexed_fixture_documents):
    """
    Pessoa sem capacete → pipeline retorna Não conforme.

    O teste verifica:
    - Pipeline não lança exceção
    - Pelo menos uma pessoa é detectada
    - Cada PersonResult tem status válido e justificativa não vazia
    - Status esperado: "Não conforme"
    - justificativa menciona capacete ou EPI ausente
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

    # Para trabalhador sem EPIs, esperamos Não conforme
    statuses = {r.status for r in response.results}
    assert "Não conforme" in statuses, (
        f"Esperado ao menos um 'Não conforme', obtidos: {statuses}\n"
        f"Justificativas: {[r.justificativa for r in response.results]}"
    )

    # A justificativa deve mencionar o problema de EPI
    for result in response.results:
        if result.status == "Não conforme":
            justificativa_lower = result.justificativa.lower()
            epi_terms = {"capacete", "epi", "equipamento", "proteção", "segurança"}
            assert any(term in justificativa_lower for term in epi_terms), (
                f"justificativa não menciona EPI ausente: {result.justificativa!r}"
            )
