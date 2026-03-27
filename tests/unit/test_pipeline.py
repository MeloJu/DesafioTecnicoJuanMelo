"""
Unit tests for the Pipeline orchestrator.

What we test:
- Happy path: vision → rag → reasoning → PipelineResponse
- correlation_id gerado internamente e passado a todos os serviços
- Edge case: nenhuma pessoa detectada → retorna resposta vazia, rag/reasoning não chamados
- Edge case: exceção em qualquer serviço → propaga, correlation_id limpo no finally
- _build_rag_query: query gerada a partir dos atributos detectados
"""
import pytest
from unittest.mock import Mock, patch

from app.pipeline.orchestrator import Pipeline, _build_rag_query
from app.schemas.output import (
    BoundingBox,
    PersonAttributes,
    PersonDetection,
    PersonResult,
    PipelineResponse,
    Rule,
)

FIXED_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_person(pessoa_id: int = 1, helmet: bool = False, vest: bool = True) -> PersonDetection:
    return PersonDetection(
        pessoa_id=pessoa_id,
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
        attributes=PersonAttributes(helmet=helmet, vest=vest),
    )


def _make_rule() -> Rule:
    return Rule(rule="Capacete obrigatório em obras.", source="manual.pdf")


def _make_result(pessoa_id: int = 1, status: str = "Conforme") -> PersonResult:
    return PersonResult(
        pessoa_id=pessoa_id,
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
        status=status,
        justificativa="Atende às regras de EPI.",
    )


def _make_pipeline(vision=None, rag=None, reasoning=None) -> Pipeline:
    return Pipeline(
        vision_service=vision or Mock(),
        rag_service=rag or Mock(),
        reasoning_service=reasoning or Mock(),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPipelineHappyPath:
    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_returns_pipeline_response(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = [_make_person()]
        rag.retrieve.return_value = [_make_rule()]
        reasoning.analyze.return_value = _make_result()

        response = _make_pipeline(vision, rag, reasoning).run("img.jpg", "Construtiva", "obras")

        assert isinstance(response, PipelineResponse)
        assert len(response.results) == 1

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_multiple_people_all_analyzed(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = [_make_person(1), _make_person(2)]
        rag.retrieve.return_value = [_make_rule()]
        reasoning.analyze.side_effect = [_make_result(1), _make_result(2)]

        response = _make_pipeline(vision, rag, reasoning).run("img.jpg", "E", "S")

        assert reasoning.analyze.call_count == 2
        assert len(response.results) == 2

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_rag_receives_empresa_and_setor(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = [_make_person()]
        rag.retrieve.return_value = [_make_rule()]
        reasoning.analyze.return_value = _make_result()

        _make_pipeline(vision, rag, reasoning).run("img.jpg", "Construtiva", "obras")

        positional = rag.retrieve.call_args[0]
        assert "Construtiva" in positional
        assert "obras" in positional


# ---------------------------------------------------------------------------
# correlation_id propagation
# ---------------------------------------------------------------------------

class TestCorrelationIdPropagation:
    def _all_args(self, call) -> list:
        """Flatten positional + keyword values into a single list."""
        return list(call[0]) + list(call[1].values())

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_correlation_id_passed_to_vision(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = [_make_person()]
        rag.retrieve.return_value = [_make_rule()]
        reasoning.analyze.return_value = _make_result()

        _make_pipeline(vision, rag, reasoning).run("img.jpg", "E", "S")

        assert FIXED_UUID in self._all_args(vision.process.call_args)

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_correlation_id_passed_to_rag(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = [_make_person()]
        rag.retrieve.return_value = [_make_rule()]
        reasoning.analyze.return_value = _make_result()

        _make_pipeline(vision, rag, reasoning).run("img.jpg", "E", "S")

        assert FIXED_UUID in self._all_args(rag.retrieve.call_args)

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_correlation_id_passed_to_reasoning(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = [_make_person()]
        rag.retrieve.return_value = [_make_rule()]
        reasoning.analyze.return_value = _make_result()

        _make_pipeline(vision, rag, reasoning).run("img.jpg", "E", "S")

        assert FIXED_UUID in self._all_args(reasoning.analyze.call_args)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_no_people_returns_empty_response(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = []

        response = _make_pipeline(vision, rag, reasoning).run("img.jpg", "E", "S")

        assert isinstance(response, PipelineResponse)
        assert response.results == []

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_no_people_does_not_call_rag_or_reasoning(self, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = []

        _make_pipeline(vision, rag, reasoning).run("img.jpg", "E", "S")

        rag.retrieve.assert_not_called()
        reasoning.analyze.assert_not_called()

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    def test_exception_propagates(self, _uuid):
        vision = Mock()
        vision.process.side_effect = RuntimeError("model unavailable")

        with pytest.raises(RuntimeError, match="model unavailable"):
            _make_pipeline(vision=vision).run("img.jpg", "E", "S")

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    @patch("app.pipeline.orchestrator.clear_correlation_id")
    def test_correlation_id_cleared_on_success(self, mock_clear, _uuid):
        vision, rag, reasoning = Mock(), Mock(), Mock()
        vision.process.return_value = [_make_person()]
        rag.retrieve.return_value = [_make_rule()]
        reasoning.analyze.return_value = _make_result()

        _make_pipeline(vision, rag, reasoning).run("img.jpg", "E", "S")

        mock_clear.assert_called_once()

    @patch("app.pipeline.orchestrator.uuid4", return_value=FIXED_UUID)
    @patch("app.pipeline.orchestrator.clear_correlation_id")
    def test_correlation_id_cleared_on_exception(self, mock_clear, _uuid):
        vision = Mock()
        vision.process.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _make_pipeline(vision=vision).run("img.jpg", "E", "S")

        mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# _build_rag_query
# ---------------------------------------------------------------------------

class TestBuildRagQuery:
    def test_absent_attributes_included_in_query(self):
        people = [_make_person(helmet=False, vest=False)]
        query = _build_rag_query(people)
        assert "capacete" in query
        assert "colete" in query

    def test_present_attributes_excluded_from_query(self):
        people = [_make_person(helmet=True, vest=True)]
        query = _build_rag_query(people)
        assert "capacete" not in query
        assert "colete" not in query

    def test_none_attributes_included_in_query(self):
        person = PersonDetection(
            pessoa_id=1,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
            attributes=PersonAttributes(),  # all None
        )
        query = _build_rag_query([person])
        assert "capacete" in query

    def test_all_present_returns_base_query(self):
        person = PersonDetection(
            pessoa_id=1,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
            attributes=PersonAttributes(helmet=True, vest=True, safety_boots=True, gloves=True),
        )
        query = _build_rag_query([person])
        assert query == "regras obrigatórias de EPI"

    def test_no_duplicate_terms_for_multiple_people(self):
        people = [_make_person(1, helmet=False), _make_person(2, helmet=False)]
        query = _build_rag_query(people)
        assert query.count("capacete") == 1
