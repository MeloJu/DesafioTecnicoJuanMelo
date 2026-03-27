from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_coords(self) -> "BoundingBox":
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")
        return self


class PersonDetection(BaseModel):
    pessoa_id: int
    bbox: BoundingBox
    attributes: Dict[str, Optional[bool]]


class Rule(BaseModel):
    rule: str
    source: str


class Chunk(BaseModel):
    text: str
    source: str
    empresa: str
    setor: str
    chunk_id: str
    metadata: dict = {}


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


ComplianceStatus = Literal["Conforme", "Não conforme", "Indeterminado"]


class PersonResult(BaseModel):
    pessoa_id: int
    bbox: BoundingBox
    status: ComplianceStatus
    justificativa: str

    @field_validator("justificativa")
    @classmethod
    def justificativa_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("justificativa cannot be empty")
        return v


class PipelineResponse(BaseModel):
    results: List[PersonResult]
