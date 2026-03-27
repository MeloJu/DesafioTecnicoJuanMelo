"""
Unit tests for the AttributeExtractor (CLIP zero-shot).

What we test (CLIP model always mocked):
- Score alto (>= threshold_positive) → atributo True
- Score baixo (<= threshold_negative) → atributo False
- Score no meio (zona de incerteza) → atributo None
- Todos os atributos EPIAttribute processados
- Recorte da bbox passado ao CLIP (não a imagem inteira)
- Thresholds customizáveis por atributo via EPIAttribute
- Imagem inválida/corrompida → dict com tudo None, sem crash
"""
import pytest
from unittest.mock import Mock
from PIL import Image

from app.vision.extractor import AttributeExtractor
from app.schemas.epi_config import EPIAttribute
from app.schemas.output import BoundingBox


CORRELATION_ID = "test-corr-id"
BBOX = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)

_DEFAULT_EPI = [
    EPIAttribute("helmet", "capacete", "wearing hard hat", "not wearing hard hat", 0.65, 0.35),
    EPIAttribute("vest", "colete", "wearing vest", "not wearing vest", 0.60, 0.40),
    EPIAttribute("safety_boots", "botas", "wearing boots", "wearing shoes", 0.70, 0.30),
    EPIAttribute("gloves", "luvas", "wearing gloves", "not wearing gloves", 0.60, 0.40),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor(scores_by_attr: dict | None = None, epi_attributes=None):
    """
    Cria um AttributeExtractor com CLIP mockado.
    scores_by_attr: {"helmet": 0.8, "vest": 0.3, ...}
    Se não fornecido, retorna 0.5 para tudo (zona de incerteza).
    """
    mock_clip = Mock()
    epis = epi_attributes or _DEFAULT_EPI

    def fake_classify(image_crop, positive_text, negative_text):
        if scores_by_attr:
            # Match por positive_text para identificar o atributo
            for epi in epis:
                if epi.clip_positive == positive_text:
                    return scores_by_attr.get(epi.name, 0.5)
        return 0.5

    mock_clip.classify.side_effect = fake_classify
    return AttributeExtractor(clip_client=mock_clip, epi_attributes=epis)


def _make_dummy_image() -> Image.Image:
    return Image.new("RGB", (200, 300), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# Zona de decisão por threshold
# ---------------------------------------------------------------------------

class TestDecisionZone:
    def test_high_score_returns_true(self):
        extractor = _make_extractor({"helmet": 0.9, "vest": 0.9, "safety_boots": 0.9, "gloves": 0.9})
        attrs = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)
        assert attrs["helmet"] is True

    def test_low_score_returns_false(self):
        extractor = _make_extractor({"helmet": 0.1, "vest": 0.1, "safety_boots": 0.1, "gloves": 0.1})
        attrs = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)
        assert attrs["helmet"] is False

    def test_mid_score_returns_none(self):
        extractor = _make_extractor({"helmet": 0.5, "vest": 0.5, "safety_boots": 0.5, "gloves": 0.5})
        attrs = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)
        assert attrs["helmet"] is None

    def test_all_attributes_classified(self):
        extractor = _make_extractor({
            "helmet": 0.9,
            "vest": 0.1,
            "safety_boots": 0.5,
            "gloves": 0.9,
        })
        attrs = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)
        assert attrs["helmet"] is True
        assert attrs["vest"] is False
        assert attrs["safety_boots"] is None
        assert attrs["gloves"] is True

    def test_returns_dict(self):
        extractor = _make_extractor()
        result = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)
        assert isinstance(result, dict)

    def test_dict_keys_match_epi_names(self):
        extractor = _make_extractor()
        result = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)
        assert set(result.keys()) == {"helmet", "vest", "safety_boots", "gloves"}


# ---------------------------------------------------------------------------
# Thresholds customizáveis via EPIAttribute
# ---------------------------------------------------------------------------

class TestCustomThresholds:
    def test_custom_positive_threshold(self):
        epi = EPIAttribute("helmet", "capacete", "wearing hard hat", "not wearing", 0.95, 0.35)
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.80  # abaixo do threshold custom
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=[epi])

        attrs = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)

        assert attrs["helmet"] is None  # 0.80 < 0.95 → zona de incerteza

    def test_custom_negative_threshold(self):
        epi = EPIAttribute("helmet", "capacete", "wearing hard hat", "not wearing", 0.65, 0.60)
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.55  # abaixo do threshold negativo custom
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=[epi])

        attrs = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)

        assert attrs["helmet"] is False  # 0.55 <= 0.60 → False


# ---------------------------------------------------------------------------
# Crop da bbox
# ---------------------------------------------------------------------------

class TestBboxCrop:
    def test_clip_called_for_each_epi_attribute(self):
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.5
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=_DEFAULT_EPI)

        extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)

        assert mock_clip.classify.call_count == 4  # helmet, vest, safety_boots, gloves

    def test_clip_receives_texts_from_epi_attribute(self):
        epi = EPIAttribute("helmet", "capacete", "pos text", "neg text", 0.65, 0.35)
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.5
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=[epi])

        extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)

        call_args = mock_clip.classify.call_args
        assert call_args[0][1] == "pos text"
        assert call_args[0][2] == "neg text"

    def test_custom_epi_list_respected(self):
        """Extractor com apenas 2 EPIs chama CLIP apenas 2 vezes."""
        epis = [
            EPIAttribute("helmet", "capacete", "pos", "neg"),
            EPIAttribute("vest", "colete", "pos", "neg"),
        ]
        mock_clip = Mock()
        mock_clip.classify.return_value = 0.5
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=epis)

        extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)

        assert mock_clip.classify.call_count == 2


# ---------------------------------------------------------------------------
# Edge case: imagem inválida
# ---------------------------------------------------------------------------

class TestImageError:
    def test_clip_exception_returns_all_none(self):
        mock_clip = Mock()
        mock_clip.classify.side_effect = Exception("CLIP model error")
        extractor = AttributeExtractor(clip_client=mock_clip, epi_attributes=_DEFAULT_EPI)

        attrs = extractor.extract(_make_dummy_image(), BBOX, CORRELATION_ID)

        assert attrs["helmet"] is None
        assert attrs["vest"] is None
        assert attrs["safety_boots"] is None
        assert attrs["gloves"] is None
