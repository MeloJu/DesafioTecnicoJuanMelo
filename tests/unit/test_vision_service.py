"""
Unit tests for VisionService.

What we test (PersonDetector and AttributeExtractor always mocked,
PIL.Image.open mocked):
- process() chama detector.detect() com image_path e correlation_id
- process() chama extractor.extract() para cada detecção
- Atributos retornados pelo extractor são inseridos na PersonDetection
- Lista vazia do detector → [] retornado, extractor não chamado
- Múltiplas pessoas → extractor chamado uma vez por pessoa
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from app.vision.service import VisionService
from app.schemas.output import BoundingBox, PersonDetection

CORRELATION_ID = "test-vision-service-id"


def _make_detection(
    pessoa_id: int = 1,
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 100.0,
    y2: float = 200.0,
) -> PersonDetection:
    return PersonDetection(
        pessoa_id=pessoa_id,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        attributes={},
    )


def _make_dummy_image() -> Image.Image:
    return Image.new("RGB", (300, 400), color=(50, 50, 50))


class TestVisionService:
    def _make_service(
        self,
        detections: list,
        attrs: dict | None = None,
    ) -> VisionService:
        mock_detector = Mock()
        mock_detector.detect.return_value = detections

        mock_extractor = Mock()
        mock_extractor.extract.return_value = attrs or {
            "helmet": True, "vest": True, "safety_boots": False, "gloves": None
        }

        return VisionService(detector=mock_detector, extractor=mock_extractor)

    def test_returns_list_with_filled_attributes(self):
        detection = _make_detection()
        service = self._make_service([detection])

        with patch("app.vision.service.Image") as mock_img:
            mock_img.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        assert len(results) == 1
        assert isinstance(results[0], PersonDetection)
        assert results[0].attributes["helmet"] is True

    def test_empty_detection_returns_empty_list(self):
        service = self._make_service([])

        with patch("app.vision.service.Image") as mock_img:
            mock_img.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        assert results == []
        service._extractor.extract.assert_not_called()

    def test_detector_receives_image_path_and_correlation_id(self):
        service = self._make_service([])

        with patch("app.vision.service.Image") as mock_img:
            mock_img.open.return_value = _make_dummy_image()
            service.process("frame_01.jpg", CORRELATION_ID)

        service._detector.detect.assert_called_once_with("frame_01.jpg", CORRELATION_ID)

    def test_extractor_called_once_per_person(self):
        detections = [_make_detection(1), _make_detection(2), _make_detection(3)]
        service = self._make_service(detections)

        with patch("app.vision.service.Image") as mock_img:
            mock_img.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        assert service._extractor.extract.call_count == 3
        assert len(results) == 3

    def test_original_detections_not_mutated(self):
        detection = _make_detection()
        service = self._make_service([detection])

        with patch("app.vision.service.Image") as mock_img:
            mock_img.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        # Original detection has empty attributes; returned copy has filled attrs
        assert detection.attributes == {}
        assert results[0].attributes["helmet"] is True

    def test_extractor_receives_correct_bbox(self):
        detection = _make_detection(x1=10, y1=20, x2=110, y2=220)
        service = self._make_service([detection])

        with patch("app.vision.service.Image") as mock_img:
            mock_img.open.return_value = _make_dummy_image()
            service.process("test.jpg", CORRELATION_ID)

        call_args = service._extractor.extract.call_args
        passed_bbox = call_args[0][1]
        assert passed_bbox.x1 == 10.0
        assert passed_bbox.y1 == 20.0
        assert passed_bbox.x2 == 110.0
        assert passed_bbox.y2 == 220.0
