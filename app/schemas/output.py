from pydantic import BaseModel
from typing import List

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class PersonResult(BaseModel):
    pessoa_id: int
    bbox: BBox
    status: str  # "Conforme", "Não conforme", "Indeterminado"
    justificativa: str

class PipelineResponse(BaseModel):
    results: List[PersonResult]