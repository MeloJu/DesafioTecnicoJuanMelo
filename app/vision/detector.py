"""
PersonDetector — detecta pessoas em imagens via YOLO (Ultralytics).

Responsabilidade única (ADR-004): encontrar bounding boxes de pessoas.
Não classifica EPIs — isso é responsabilidade do IAttributeExtractor.

Retorna List[PersonDetection] com attributes todos None.
O IAttributeExtractor preenche os atributos em seguida.

Classe YOLO (cls=0 no COCO) é o único índice considerado pessoa.
"""
from typing import List

from ultralytics import YOLO

from app.logging.logger import get_logger
from app.schemas.output import BoundingBox, PersonAttributes, PersonDetection

log = get_logger()

_PERSON_CLASS_ID = 0


class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self._model = YOLO(model_path)

    def detect(self, image_path: str, correlation_id: str) -> List[PersonDetection]:
        yolo_results = self._model(image_path)

        detections: List[PersonDetection] = []
        pessoa_id = 1

        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != _PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                detections.append(PersonDetection(
                    pessoa_id=pessoa_id,
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                    attributes=PersonAttributes(),
                ))
                pessoa_id += 1

        log.info(
            "vision_processed",
            correlation_id=correlation_id,
            image_path=image_path,
            total_people=len(detections),
        )
        return detections
