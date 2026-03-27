"""
RagService — fachada exposta ao pipeline (ADR-005).

Responsabilidade: dado (empresa, setor, query), retornar List[Rule].
Delega o retrieval semântico ao EmbeddingService.

O nome da collection no ChromaDB é derivado do nome da empresa
(lowercase, caracteres especiais → underscore).
"""
import re
from typing import List

from app.logging.logger import get_logger
from app.rag.embedding_service import EmbeddingService
from app.schemas.output import Rule

log = get_logger()

_TOP_K = 5


def _collection_name(empresa: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", empresa.lower()).strip("_")


class RagService:
    def __init__(self, embedding_service: EmbeddingService):
        self._embedder = embedding_service

    def retrieve(
        self,
        empresa: str,
        setor: str,
        query: str,
        correlation_id: str,
    ) -> List[Rule]:
        collection = _collection_name(empresa)
        scored_chunks = self._embedder.query(query, collection, top_k=_TOP_K)
        rules = [Rule(rule=sc.chunk.text, source=sc.chunk.source) for sc in scored_chunks]

        if not rules:
            log.warning(
                "no_rules_found",
                correlation_id=correlation_id,
                empresa=empresa,
                setor=setor,
            )
        else:
            log.info(
                "rules_retrieved",
                correlation_id=correlation_id,
                empresa=empresa,
                setor=setor,
                total=len(rules),
            )

        return rules
