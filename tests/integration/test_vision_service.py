"""
Integration test: PersonDetector + AttributeExtractor + VisionService.

Módulos reais integrados. Modelos externos (YOLO, CLIP) mockados.
Verifica que a fachada VisionService orquestra corretamente os dois
sub-módulos e retorna PersonDetection com attributes preenchidos.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from app.vision.service import VisionService
from app.vision.detector import PersonDetector
from app.vision.extractor import AttributeExtractor
from app.schemas.epi_config import EPIAttribute
from app.schemas.output import PersonDetection

CORRELATION_ID = "integration-corr-id"

_DEFAULT_EPI = [
    EPIAttribute("helmet", "capacete", "wearing hard hat", "not wearing hard hat", 0.65, 0.35),
    EPIAttribute("vest", "colete", "wearing vest", "not wearing vest", 0.60, 0.40),
    EPIAttribute("safety_boots", "botas", "wearing boots", "wearing shoes", 0.70, 0.30),
    EPIAttribute("gloves", "luvas", "wearing gloves", "not wearing gloves", 0.60, 0.40),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_box(x1=10.0, y1=20.0, x2=110.0, y2=220.0, cls=0):
    box = Mock()
    box.xyxy = [[x1, y1, x2, y2]]
    box.cls = [cls]
    box.conf = [0.92]
    return box


def _make_yolo_result(boxes):
    result = Mock()
    result.boxes = boxes
    return result


def _make_dummy_image():
    return Image.new("RGB", (300, 400), color=(100, 100, 100))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestVisionServiceIntegration:

    @patch("app.vision.detector.YOLO")
    def test_attributes_filled_after_detection(self, mock_yolo_cls):
        """PersonDetection sai do detector com attributes={};
        VisionService deve preenchê-los via AttributeExtractor."""
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([_make_yolo_box()])]
        mock_yolo_cls.return_value = mock_model

        mock_clip = Mock()
        mock_clip.classify.return_value = 0.9  # acima do threshold → True

        detector = PersonDetector(model_path="yolov8n.pt")
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=_DEFAULT_EPI)
        service = VisionService(detector=detector, extractor=extractor)

        with patch("app.vision.service.Image") as mock_image_module:
            mock_image_module.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        assert len(results) == 1
        assert isinstance(results[0], PersonDetection)
        assert results[0].attributes["helmet"] is True

    @patch("app.vision.detector.YOLO")
    def test_attributes_reflect_clip_scores(self, mock_yolo_cls):
        """Score baixo do CLIP resulta em False no atributo."""
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([_make_yolo_box()])]
        mock_yolo_cls.return_value = mock_model

        mock_clip = Mock()
        mock_clip.classify.return_value = 0.1  # abaixo do threshold → False

        detector = PersonDetector(model_path="yolov8n.pt")
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=_DEFAULT_EPI)
        service = VisionService(detector=detector, extractor=extractor)

        with patch("app.vision.service.Image") as mock_image_module:
            mock_image_module.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        assert results[0].attributes["helmet"] is False

    @patch("app.vision.detector.YOLO")
    def test_no_people_returns_empty(self, mock_yolo_cls):
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([])]
        mock_yolo_cls.return_value = mock_model

        mock_clip = Mock()
        detector = PersonDetector(model_path="yolov8n.pt")
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=_DEFAULT_EPI)
        service = VisionService(detector=detector, extractor=extractor)

        with patch("app.vision.service.Image") as mock_image_module:
            mock_image_module.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        assert results == []
        mock_clip.classify.assert_not_called()

    @patch("app.vision.detector.YOLO")
    def test_multiple_people_each_gets_attributes(self, mock_yolo_cls):
        boxes = [_make_yolo_box(0, 0, 50, 100), _make_yolo_box(60, 0, 120, 100)]
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result(boxes)]
        mock_yolo_cls.return_value = mock_model

        mock_clip = Mock()
        mock_clip.classify.return_value = 0.5  # zona de incerteza → None

        detector = PersonDetector(model_path="yolov8n.pt")
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=_DEFAULT_EPI)
        service = VisionService(detector=detector, extractor=extractor)

        with patch("app.vision.service.Image") as mock_image_module:
            mock_image_module.open.return_value = _make_dummy_image()
            results = service.process("test.jpg", CORRELATION_ID)

        assert len(results) == 2
        # CLIP chamado 4 atributos × 2 pessoas = 8 vezes
        assert mock_clip.classify.call_count == 8
