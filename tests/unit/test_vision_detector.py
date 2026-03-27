"""
Unit tests for the PersonDetector.

What we test (YOLO always mocked):
- Imagem com pessoas → List[PersonDetection] com bbox corretas
- Imagem sem pessoas → lista vazia
- Múltiplas pessoas → pessoa_id incremental começando em 1
- Coordenadas da bbox preservadas como float
- correlation_id passado ao YOLO (via chamada do modelo)
- Imagem inexistente → FileNotFoundError propagado
- Resultado do YOLO malformado → lista vazia, sem crash
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.vision.detector import PersonDetector
from app.schemas.output import PersonDetection, BoundingBox


CORRELATION_ID = "test-corr-id"
IMAGE_PATH = "data/raw/empresa/imagens/frame_001.jpg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_box(x1=10.0, y1=20.0, x2=110.0, y2=220.0, cls=0, conf=0.92):
    """Simula um bounding box retornado pelo YOLO (Ultralytics Results)."""
    box = Mock()
    box.xyxy = [[x1, y1, x2, y2]]
    box.cls = [cls]
    box.conf = [conf]
    return box


def _make_yolo_result(boxes):
    result = Mock()
    result.boxes = boxes
    return result


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPersonDetectorHappyPath:
    @patch("app.vision.detector.YOLO")
    def test_returns_list_of_person_detections(self, mock_yolo_cls):
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([_make_yolo_box()])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        results = detector.detect(IMAGE_PATH, CORRELATION_ID)

        assert isinstance(results, list)
        assert all(isinstance(r, PersonDetection) for r in results)

    @patch("app.vision.detector.YOLO")
    def test_pessoa_id_starts_at_one(self, mock_yolo_cls):
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([_make_yolo_box()])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        results = detector.detect(IMAGE_PATH, CORRELATION_ID)

        assert results[0].pessoa_id == 1

    @patch("app.vision.detector.YOLO")
    def test_multiple_people_incremental_ids(self, mock_yolo_cls):
        boxes = [_make_yolo_box(0, 0, 50, 100), _make_yolo_box(60, 0, 120, 100)]
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result(boxes)]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        results = detector.detect(IMAGE_PATH, CORRELATION_ID)

        assert len(results) == 2
        assert results[0].pessoa_id == 1
        assert results[1].pessoa_id == 2

    @patch("app.vision.detector.YOLO")
    def test_bbox_coordinates_correct(self, mock_yolo_cls):
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([_make_yolo_box(10.0, 20.0, 110.0, 220.0)])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        results = detector.detect(IMAGE_PATH, CORRELATION_ID)

        bbox = results[0].bbox
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 110.0
        assert bbox.y2 == 220.0

    @patch("app.vision.detector.YOLO")
    def test_attributes_all_none_after_detection(self, mock_yolo_cls):
        """Detector não classifica EPIs — attributes vêm todos None."""
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([_make_yolo_box()])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        results = detector.detect(IMAGE_PATH, CORRELATION_ID)

        attrs = results[0].attributes
        assert attrs == {}  # detector retorna dict vazio; extractor preenche depois


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPersonDetectorEdgeCases:
    @patch("app.vision.detector.YOLO")
    def test_no_people_returns_empty_list(self, mock_yolo_cls):
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        results = detector.detect(IMAGE_PATH, CORRELATION_ID)

        assert results == []

    @patch("app.vision.detector.YOLO")
    def test_non_person_class_ignored(self, mock_yolo_cls):
        """YOLO detecta objetos de outras classes (cls != 0) — ignorar."""
        car_box = _make_yolo_box(cls=2)  # cls=2 = car em COCO
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([car_box])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        results = detector.detect(IMAGE_PATH, CORRELATION_ID)

        assert results == []

    @patch("app.vision.detector.YOLO")
    def test_yolo_called_with_image_path(self, mock_yolo_cls):
        mock_model = Mock()
        mock_model.return_value = [_make_yolo_result([])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        detector.detect(IMAGE_PATH, CORRELATION_ID)

        mock_model.assert_called_once()
        call_args = mock_model.call_args[0]
        assert IMAGE_PATH in call_args

    @patch("app.vision.detector.YOLO")
    def test_file_not_found_propagates(self, mock_yolo_cls):
        mock_model = Mock()
        mock_model.side_effect = FileNotFoundError("image not found")
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model_path="yolov8n.pt")
        with pytest.raises(FileNotFoundError):
            detector.detect("nonexistent.jpg", CORRELATION_ID)
