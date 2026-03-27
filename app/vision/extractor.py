"""
AttributeExtractor — classifica EPIs via CLIP zero-shot (ADR-003b, ADR-009).

Para cada EPIAttribute na lista recebida, compara o recorte da pessoa contra
os textos CLIP positivo e negativo e aplica a zona de incerteza:

  score >= threshold_positive  →  True   (EPI presente)
  score <= threshold_negative  →  False  (EPI ausente)
  entre os dois                →  None   (incerteza → Indeterminado no reasoning)

Recebe List[EPIAttribute] via DI (ADR-006) — sem atributos hardcodados.
clip_client injetado via DI; mock em testes, CLIPClient real em produção.
"""
from typing import Dict, List, Optional

from PIL import Image

from app.logging.logger import get_logger
from app.schemas.epi_config import EPIAttribute
from app.schemas.output import BoundingBox

log = get_logger()


class AttributeExtractor:
    def __init__(self, clip_client, epi_attributes: List[EPIAttribute]):
        self._clip = clip_client
        self._epi_attributes = epi_attributes

    def extract(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        correlation_id: str,
    ) -> Dict[str, Optional[bool]]:
        crop = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        results: Dict[str, Optional[bool]] = {}

        for epi in self._epi_attributes:
            try:
                score = self._clip.classify(crop, epi.clip_positive, epi.clip_negative)
                results[epi.name] = self._apply_threshold(score, epi)
            except Exception as exc:
                log.warning(
                    "clip_classification_failed",
                    correlation_id=correlation_id,
                    attribute=epi.name,
                    error=str(exc),
                )
                results[epi.name] = None

        return results

    def _apply_threshold(self, score: float, epi: EPIAttribute) -> Optional[bool]:
        if score >= epi.threshold_positive:
            return True
        if score <= epi.threshold_negative:
            return False
        return None
