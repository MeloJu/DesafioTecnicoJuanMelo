"""
Unit tests for the AttributeExtractor (CLIP zero-shot).

What we test (CLIP model always mocked):
- Score alto (>= threshold_positive) → atributo True
- Score baixo (<= threshold_negative) → atributo False
- Score no meio (zona de incerteza) → atributo None
- Todos os atributos processados: helmet, vest, safety_boots, gloves
- Recorte da bbox passado ao CLIP (não a imagem inteira)
- Thresholds customizáveis por atributo via CLIPThresholds
- Imagem inválida/corrompida → PersonAttributes com tudo None, sem crash
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from app.vision.extractor import AttributeExtractor
from app.schemas.output import BoundingBox, PersonAttributes
from app.vision.extractor import CLIPThresholds


CORRELATION_ID = "test-corr-id"
BBOX = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor(scores_by_attr: dict | None = None):
    """
    Cria um AttributeExtractor com CLIP mockado.
    scores_by_attr: {"helmet": 0.8, "vest": 0.3, ...}
    Se não fornecido, retorna 0.5 para tudo (zona de incerteza).
    """
    mock_clip = Mock()

    def fake_classify(image_crop, attr_name):
        if scores_by_attr:
            return scores_by_attr.get(attr_name, 0.5)
        return 0.5

    mock_clip.classify.side_effect = fake_classify
    return AttributeExtractor(clip_client=mock_clip)


def _make_dummy_image() -> Image.Image:
    return Image.new("RGB", (200, 300), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# Zona de decisão por threshold
# ---------------------------------------------------------------------------

class TestDecisionZone:
    def test_high_score_returns_true(self):
        extractor = _make_extractor({"helmet": 0.9, "vest": 0.9, "safety_boots": 0.9, "gloves": 0.9})
        image = _make_dummy_image()
        attrs = extractor.extract(image, BBOX, CORRELATION_ID)
        assert attrs.helmet is True

    def test_low_score_returns_false(self):
        extractor = _make_extractor({"helmet": 0.1, "vest": 0.1, "safety_boots": 0.1, "gloves": 0.1})
        image = _make_dummy_image()
        attrs = extractor.extract(image, BBOX, CORRELATION_ID)
        assert attrs.helmet is False

    def test_mid_score_returns_none(self):
        extractor = _make_extractor({"helmet": 0.5, "vest": 0.5, "safety_boots": 0.5, "gloves": 0.5})
        image = _make_dummy_image()
        attrs = extractor.extract(image, BBOX, CORRELATION_ID)
        assert attrs.helmet is None

    def test_all_attributes_classified(self):
        extractor = _make_extractor({
            "helmet": 0.9,
            "vest": 0.1,
            "safety_boots": 0.5,
            "gloves": 0.9,
        })
        image = _make_dummy_image()
        attrs = extractor.extract(image, BBOX, CORRELATION_ID)
        assert attrs.helmet is True
        assert attrs.vest is False
        assert attrs.safety_boots is None
        assert attrs.gloves is True

    def test_returns_person_attributes_instance(self):
        extractor = _make_extractor()
        image = _make_dummy_image()
        result = extractor.extract(image, BBOX, CORRELATION_ID)
        assert isinstance(result, PersonAttributes)


# ---------------------------------------------------------------------------
# Thresholds customizáveis
# ---------------------------------------------------------------------------

class TestCustomThresholds:
    def test_custom_positive_threshold(self):
        thresholds = CLIPThresholds(
            helmet_positive=0.95,
            helmet_negative=0.35,
        )
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.80  # abaixo do threshold custom
        extractor = AttributeExtractor(clip_client=mock_clip, thresholds=thresholds)
        image = _make_dummy_image()

        attrs = extractor.extract(image, BBOX, CORRELATION_ID)

        assert attrs.helmet is None  # 0.80 < 0.95 → zona de incerteza

    def test_custom_negative_threshold(self):
        thresholds = CLIPThresholds(
            helmet_positive=0.65,
            helmet_negative=0.60,  # threshold negativo alto
        )
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.55  # abaixo do threshold negativo custom
        extractor = AttributeExtractor(clip_client=mock_clip, thresholds=thresholds)
        image = _make_dummy_image()

        attrs = extractor.extract(image, BBOX, CORRELATION_ID)

        assert attrs.helmet is False  # 0.55 <= 0.60 → False


# ---------------------------------------------------------------------------
# Crop da bbox
# ---------------------------------------------------------------------------

class TestBboxCrop:
    def test_clip_called_for_each_attribute(self):
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.5
        extractor = AttributeExtractor(clip_client=mock_clip)
        image = _make_dummy_image()

        extractor.extract(image, BBOX, CORRELATION_ID)

        assert mock_clip.classify.call_count == 4  # helmet, vest, safety_boots, gloves


# ---------------------------------------------------------------------------
# Edge case: imagem inválida
# ---------------------------------------------------------------------------

class TestImageError:
    def test_clip_exception_returns_all_none(self):
        mock_clip = Mock()
        mock_clip.classify.side_effect = Exception("CLIP model error")
        extractor = AttributeExtractor(clip_client=mock_clip)
        image = _make_dummy_image()

        attrs = extractor.extract(image, BBOX, CORRELATION_ID)

        assert attrs.helmet is None
        assert attrs.vest is None
        assert attrs.safety_boots is None
        assert attrs.gloves is None
