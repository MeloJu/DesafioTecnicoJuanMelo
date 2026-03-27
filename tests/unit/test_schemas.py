import pytest
from pydantic import ValidationError

from app.schemas.output import (
    BoundingBox,
    PersonDetection,
    Rule,
    Chunk,
    ScoredChunk,
    PersonResult,
    PipelineResponse,
)


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_valid(self):
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0)
        assert bbox.x1 == 0.0
        assert bbox.y2 == 200.0

    def test_integer_coords_coerced_to_float(self):
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        assert isinstance(bbox.x1, float)

    def test_x2_must_be_greater_than_x1(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=100.0, y1=0.0, x2=50.0, y2=100.0)

    def test_y2_must_be_greater_than_y1(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=0.0, y1=100.0, x2=100.0, y2=50.0)

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=0.0, y1=0.0, x2=100.0)  # missing y2


# ---------------------------------------------------------------------------
# PersonDetection (attributes now a dict)
# ---------------------------------------------------------------------------

class TestPersonDetection:
    def test_valid_with_dict_attributes(self):
        detection = PersonDetection(
            pessoa_id=1,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
            attributes={"helmet": True, "vest": False},
        )
        assert detection.pessoa_id == 1
        assert detection.attributes["helmet"] is True
        assert detection.attributes["vest"] is False

    def test_empty_dict_attributes_allowed(self):
        detection = PersonDetection(
            pessoa_id=1,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
            attributes={},
        )
        assert detection.attributes == {}

    def test_none_values_in_attributes(self):
        detection = PersonDetection(
            pessoa_id=1,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
            attributes={"helmet": None, "vest": True},
        )
        assert detection.attributes["helmet"] is None
        assert detection.attributes["vest"] is True

    def test_missing_pessoa_id_raises(self):
        with pytest.raises(ValidationError):
            PersonDetection(
                bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
                attributes={},
            )

    def test_missing_attributes_raises(self):
        with pytest.raises(ValidationError):
            PersonDetection(
                pessoa_id=1,
                bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
            )


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

class TestRule:
    def test_valid(self):
        rule = Rule(rule="Uso de capacete obrigatório", source="manual_construtiva.pdf")
        assert rule.rule == "Uso de capacete obrigatório"
        assert rule.source == "manual_construtiva.pdf"

    def test_missing_rule_raises(self):
        with pytest.raises(ValidationError):
            Rule(source="manual.pdf")

    def test_missing_source_raises(self):
        with pytest.raises(ValidationError):
            Rule(rule="Uso de capacete obrigatório")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_valid_with_default_metadata(self):
        chunk = Chunk(
            text="Capacete obrigatório em obras.",
            source="manual.pdf",
            empresa="Construtiva",
            setor="obras",
            chunk_id="abc-123",
        )
        assert chunk.metadata == {}

    def test_custom_metadata(self):
        chunk = Chunk(
            text="x",
            source="f.pdf",
            empresa="E",
            setor="S",
            chunk_id="1",
            metadata={"page": 3, "section": "2.1"},
        )
        assert chunk.metadata["page"] == 3

    def test_metadata_is_independent_per_instance(self):
        c1 = Chunk(text="a", source="f.pdf", empresa="E", setor="S", chunk_id="1")
        c2 = Chunk(text="b", source="f.pdf", empresa="E", setor="S", chunk_id="2")
        c1.metadata["x"] = 1
        assert "x" not in c2.metadata

    def test_missing_chunk_id_raises(self):
        with pytest.raises(ValidationError):
            Chunk(text="x", source="f.pdf", empresa="E", setor="S")

    def test_missing_empresa_raises(self):
        with pytest.raises(ValidationError):
            Chunk(text="x", source="f.pdf", setor="S", chunk_id="1")


# ---------------------------------------------------------------------------
# ScoredChunk
# ---------------------------------------------------------------------------

class TestScoredChunk:
    def _make_chunk(self) -> Chunk:
        return Chunk(text="x", source="f.pdf", empresa="E", setor="S", chunk_id="1")

    def test_valid(self):
        sc = ScoredChunk(chunk=self._make_chunk(), score=0.87)
        assert sc.score == 0.87

    def test_score_zero(self):
        sc = ScoredChunk(chunk=self._make_chunk(), score=0.0)
        assert sc.score == 0.0

    def test_missing_score_raises(self):
        with pytest.raises(ValidationError):
            ScoredChunk(chunk=self._make_chunk())

    def test_missing_chunk_raises(self):
        with pytest.raises(ValidationError):
            ScoredChunk(score=0.5)


# ---------------------------------------------------------------------------
# PersonResult
# ---------------------------------------------------------------------------

class TestPersonResult:
    def _make_bbox(self) -> BoundingBox:
        return BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0)

    def test_valid_conforme(self):
        result = PersonResult(
            pessoa_id=1,
            bbox=self._make_bbox(),
            status="Conforme",
            justificativa="Todos os EPIs presentes.",
        )
        assert result.status == "Conforme"

    def test_valid_nao_conforme(self):
        result = PersonResult(
            pessoa_id=2,
            bbox=self._make_bbox(),
            status="Não conforme",
            justificativa="Capacete ausente conforme regra 3.1.",
        )
        assert result.status == "Não conforme"

    def test_valid_indeterminado(self):
        result = PersonResult(
            pessoa_id=3,
            bbox=self._make_bbox(),
            status="Indeterminado",
            justificativa="Score CLIP na zona de incerteza (0.48).",
        )
        assert result.status == "Indeterminado"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            PersonResult(
                pessoa_id=1,
                bbox=self._make_bbox(),
                status="Aprovado",
                justificativa="...",
            )

    def test_empty_justificativa_raises(self):
        with pytest.raises(ValidationError):
            PersonResult(
                pessoa_id=1,
                bbox=self._make_bbox(),
                status="Conforme",
                justificativa="",
            )

    def test_whitespace_only_justificativa_raises(self):
        with pytest.raises(ValidationError):
            PersonResult(
                pessoa_id=1,
                bbox=self._make_bbox(),
                status="Conforme",
                justificativa="   ",
            )

    def test_missing_justificativa_raises(self):
        with pytest.raises(ValidationError):
            PersonResult(pessoa_id=1, bbox=self._make_bbox(), status="Conforme")


# ---------------------------------------------------------------------------
# PipelineResponse
# ---------------------------------------------------------------------------

class TestPipelineResponse:
    def _make_result(self, pessoa_id: int = 1) -> PersonResult:
        return PersonResult(
            pessoa_id=pessoa_id,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
            status="Conforme",
            justificativa="OK.",
        )

    def test_valid_with_results(self):
        response = PipelineResponse(results=[self._make_result()])
        assert len(response.results) == 1

    def test_multiple_results(self):
        response = PipelineResponse(results=[self._make_result(1), self._make_result(2)])
        assert len(response.results) == 2
        assert response.results[1].pessoa_id == 2

    def test_empty_results_allowed(self):
        response = PipelineResponse(results=[])
        assert response.results == []

    def test_invalid_item_type_raises(self):
        with pytest.raises(ValidationError):
            PipelineResponse(results=["not a PersonResult"])

    def test_missing_results_raises(self):
        with pytest.raises(ValidationError):
            PipelineResponse()
