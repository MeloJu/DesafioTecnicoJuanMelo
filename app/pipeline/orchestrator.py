from typing import List
from uuid import uuid4

from app.logging.logger import bind_correlation_id, clear_correlation_id, get_logger
from app.schemas.epi_config import EPIAttribute
from app.schemas.output import PersonDetection, PipelineResponse

log = get_logger()


class Pipeline:
    def __init__(self, vision_service, rag_service, reasoning_service, epi_attributes: List[EPIAttribute]):
        self._vision = vision_service
        self._rag = rag_service
        self._reasoning = reasoning_service
        self._epi_attributes = epi_attributes

    def run(self, image_path: str, empresa: str, setor: str) -> PipelineResponse:
        correlation_id = str(uuid4())
        bind_correlation_id(correlation_id)
        log.info("pipeline_started", image_path=image_path, empresa=empresa, setor=setor)

        try:
            people = self._vision.process(image_path, correlation_id)

            if not people:
                log.warning("no_people_detected", image_path=image_path)
                return PipelineResponse(results=[])

            query = _build_rag_query(people, self._epi_attributes)
            rules = self._rag.retrieve(empresa, setor, query, correlation_id)
            results = [
                self._reasoning.analyze(person, rules, correlation_id, self._epi_attributes, empresa)
                for person in people
            ]

            log.info("pipeline_completed", total_people=len(results))
            return PipelineResponse(results=results)

        except Exception as exc:
            log.error("pipeline_failed", error=str(exc))
            raise

        finally:
            clear_correlation_id()


def _build_rag_query(people: List[PersonDetection], epi_attributes: List[EPIAttribute]) -> str:
    """Build a semantic search query from detected person attributes.

    Attributes absent (False) or uncertain (None) are included so the RAG
    retrieves the relevant rules for verification. Attributes confirmed
    present (True) are excluded — no need to look up rules for them.
    """
    seen: set = set()
    terms: List[str] = []
    for person in people:
        for epi in epi_attributes:
            if person.attributes.get(epi.name) is not True and epi.label_pt not in seen:
                terms.append(epi.label_pt)
                seen.add(epi.label_pt)

    base = "regras obrigatórias de EPI"
    return f"{base}: {' '.join(terms)}" if terms else base
