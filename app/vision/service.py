"""
VisionService — fachada exposta ao pipeline (ADR-004).

Orquestra PersonDetector (YOLO) + AttributeExtractor (CLIP):
  1. Detecta pessoas → List[PersonDetection] com attributes=None
  2. Para cada pessoa, extrai atributos → PersonAttributes preenchida
  3. Retorna lista com PersonDetection completas (bbox + attributes)

O pipeline chama apenas process() — não conhece detector nem extractor.
"""
from typing import List

from PIL import Image

from app.logging.logger import get_logger
from app.schemas.output import PersonDetection
from app.vision.detector import PersonDetector
from app.vision.extractor import AttributeExtractor

log = get_logger()


class VisionService:
    def __init__(self, detector: PersonDetector, extractor: AttributeExtractor):
        self._detector = detector
        self._extractor = extractor

    def process(self, image_path: str, correlation_id: str) -> List[PersonDetection]:
        image = Image.open(image_path)
        detections = self._detector.detect(image_path, correlation_id)

        result = []
        for detection in detections:
            attrs = self._extractor.extract(image, detection.bbox, correlation_id)
            result.append(detection.model_copy(update={"attributes": attrs}))

        return result
