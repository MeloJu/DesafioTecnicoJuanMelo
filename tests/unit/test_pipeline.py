from unittest.mock import Mock

from app.pipeline.orchestrator import Pipeline
from app.schemas.output import BBox, PersonResult, PipelineResponse


def test_pipeline_run_orchestrates_services_and_returns_pipeline_response():
    vision_service = Mock()
    rag_service = Mock()
    reasoning_service = Mock()

    detected_people = [
        {"pessoa_id": 1, "bbox": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}},
        {"pessoa_id": 2, "bbox": {"x1": 50, "y1": 60, "x2": 70, "y2": 80}},
    ]
    rules = ["Regra 1", "Regra 2"]

    vision_service.process.return_value = detected_people
    rag_service.retrieve.return_value = rules

    reasoning_service.analyze.side_effect = [
        PersonResult(
            pessoa_id=1,
            bbox=BBox(x1=10, y1=20, x2=30, y2=40),
            status="Conforme",
            justificativa="Atende às regras.",
        ),
        PersonResult(
            pessoa_id=2,
            bbox=BBox(x1=50, y1=60, x2=70, y2=80),
            status="Não conforme",
            justificativa="Violou a Regra 2.",
        ),
    ]

    pipeline = Pipeline(
        vision_service=vision_service,
        rag_service=rag_service,
        reasoning_service=reasoning_service,
    )

    response = pipeline.run(
        image_path="data/raw/empresa_x/imagens/frame_001.jpg",
        empresa="Empresa X",
        setor="Operação",
    )

    vision_service.process.assert_called_once_with("data/raw/empresa_x/imagens/frame_001.jpg")
    rag_service.retrieve.assert_called_once_with("Empresa X", "Operação")

    assert reasoning_service.analyze.call_count == 2
    reasoning_service.analyze.assert_any_call(detected_people[0], rules)
    reasoning_service.analyze.assert_any_call(detected_people[1], rules)

    assert isinstance(response, PipelineResponse)
    assert len(response.results) == 2
    assert all(isinstance(item, PersonResult) for item in response.results)
    assert response.results[0].pessoa_id == 1
    assert response.results[1].status == "Não conforme"
