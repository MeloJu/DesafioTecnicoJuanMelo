"""
AttributeExtractor — classifica EPIs via CLIP zero-shot (ADR-003b, ADR-009).

Para cada atributo, compara o recorte da pessoa contra dois textos opostos
e aplica a zona de incerteza:

  score >= threshold_positive  →  True   (EPI presente)
  score <= threshold_negative  →  False  (EPI ausente)
  entre os dois                →  None   (incerteza → Indeterminado no reasoning)

Thresholds configuráveis por atributo via CLIPThresholds (OCP — ADR-009).
clip_client injetado via DI (ADR-006); mock em testes, CLIPClient real em produção.
"""
from typing import Optional
from PIL import Image

from pydantic_settings import BaseSettings

from app.logging.logger import get_logger
from app.schemas.output import BoundingBox, PersonAttributes

log = get_logger()


class CLIPThresholds(BaseSettings):
    helmet_positive: float = 0.65
    helmet_negative: float = 0.35
    vest_positive: float = 0.60
    vest_negative: float = 0.40
    safety_boots_positive: float = 0.70
    safety_boots_negative: float = 0.30
    gloves_positive: float = 0.60
    gloves_negative: float = 0.40

    model_config = {"env_prefix": "CLIP_"}


_ATTRIBUTES = ["helmet", "vest", "safety_boots", "gloves"]


class AttributeExtractor:
    def __init__(self, clip_client, thresholds: CLIPThresholds = None):
        self._clip = clip_client
        self._thresholds = thresholds or CLIPThresholds()

    def extract(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        correlation_id: str,
    ) -> PersonAttributes:
        crop = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        results = {}

        for attr in _ATTRIBUTES:
            try:
                score = self._clip.classify(crop, attr)
                results[attr] = self._apply_threshold(score, attr)
            except Exception as exc:
                log.warning(
                    "clip_classification_failed",
                    correlation_id=correlation_id,
                    attribute=attr,
                    error=str(exc),
                )
                results[attr] = None

        return PersonAttributes(**results)

    def _apply_threshold(self, score: float, attr: str) -> Optional[bool]:
        t = self._thresholds
        pos = getattr(t, f"{attr}_positive")
        neg = getattr(t, f"{attr}_negative")

        if score >= pos:
            return True
        if score <= neg:
            return False
        return None
