"""
ReasoningService — analisa conformidade de EPI via Llama (Ollama).

Fluxo:
  1. Sem regras → Indeterminado imediatamente (sem chamar o LLM)
  2. Monta prompt com atributos detectados + regras recuperadas
  3. Chama Llama via llm_client.generate()
  4. Parseia JSON da resposta → PersonResult
  5. Qualquer falha (JSON inválido, status inválido, exceção) → Indeterminado
     com justificativa explicando o motivo — nunca silencia o erro

O llm_client é injetado no __init__ (DI manual, ADR-006).
Em produção: OllamaClient. Em testes unitários: sempre mockado.
"""
import json
from typing import List

from app.logging.logger import get_logger
from app.schemas.output import BoundingBox, PersonDetection, PersonResult, Rule

log = get_logger()

_VALID_STATUSES = {"Conforme", "Não conforme", "Indeterminado"}

_PROMPT_TEMPLATE = """\
Você é um sistema de verificação de conformidade de EPIs.

Atributos detectados na pessoa:
{attributes}

Regras aplicáveis:
{rules}

Analise se a pessoa está em conformidade com as regras e responda APENAS com JSON válido:
{{"status": "<Conforme|Não conforme|Indeterminado>", "justificativa": "<explicação objetiva citando a regra>"}}
"""


class ReasoningService:
    def __init__(self, llm_client):
        self._llm = llm_client

    def analyze(
        self,
        person: PersonDetection,
        rules: List[Rule],
        correlation_id: str,
    ) -> PersonResult:
        if not rules:
            log.warning(
                "no_rules_found",
                correlation_id=correlation_id,
                pessoa_id=person.pessoa_id,
            )
            return self._indeterminado(person, "Nenhuma regra encontrada para esta empresa e setor.")

        prompt = _build_prompt(person, rules)

        try:
            raw = self._llm.generate(prompt, correlation_id=correlation_id)
            return self._parse_response(raw, person, correlation_id)

        except Exception as exc:
            log.error(
                "llm_call_failed",
                correlation_id=correlation_id,
                pessoa_id=person.pessoa_id,
                error=str(exc),
            )
            return self._indeterminado(
                person,
                f"Falha na chamada ao modelo de linguagem: {exc}",
            )

    def _parse_response(
        self,
        raw: str,
        person: PersonDetection,
        correlation_id: str,
    ) -> PersonResult:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            log.error(
                "llm_call_failed",
                correlation_id=correlation_id,
                pessoa_id=person.pessoa_id,
                error=f"JSON inválido: {exc}",
            )
            return self._indeterminado(person, f"Resposta do modelo não é JSON válido: {raw[:80]}")

        status = data.get("status", "")
        justificativa = data.get("justificativa", "").strip()

        if status not in _VALID_STATUSES:
            log.error(
                "llm_call_failed",
                correlation_id=correlation_id,
                pessoa_id=person.pessoa_id,
                error=f"Status inválido recebido: '{status}'",
            )
            return self._indeterminado(person, f"Status inválido retornado pelo modelo: '{status}'")

        if not justificativa:
            log.error(
                "llm_call_failed",
                correlation_id=correlation_id,
                pessoa_id=person.pessoa_id,
                error="justificativa vazia na resposta do modelo",
            )
            return self._indeterminado(person, "Modelo retornou justificativa vazia.")

        return PersonResult(
            pessoa_id=person.pessoa_id,
            bbox=person.bbox,
            status=status,
            justificativa=justificativa,
        )

    @staticmethod
    def _indeterminado(person: PersonDetection, justificativa: str) -> PersonResult:
        return PersonResult(
            pessoa_id=person.pessoa_id,
            bbox=person.bbox,
            status="Indeterminado",
            justificativa=justificativa,
        )


def _build_prompt(person: PersonDetection, rules: List[Rule]) -> str:
    attrs = person.attributes
    attribute_lines = "\n".join([
        f"- capacete: {'presente' if attrs.helmet is True else 'ausente' if attrs.helmet is False else 'incerto'}",
        f"- colete: {'presente' if attrs.vest is True else 'ausente' if attrs.vest is False else 'incerto'}",
        f"- botas de segurança: {'presente' if attrs.safety_boots is True else 'ausente' if attrs.safety_boots is False else 'incerto'}",
        f"- luvas: {'presente' if attrs.gloves is True else 'ausente' if attrs.gloves is False else 'incerto'}",
    ])
    rule_lines = "\n".join(f"- {r.rule} (fonte: {r.source})" for r in rules)
    return _PROMPT_TEMPLATE.format(attributes=attribute_lines, rules=rule_lines)
