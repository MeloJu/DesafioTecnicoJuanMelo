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
import re
from typing import Dict, List, Optional

from app.logging.logger import get_logger
from app.schemas.epi_config import EPIAttribute
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
        epi_attributes: List[EPIAttribute] = None,
    ) -> PersonResult:
        if not rules:
            log.warning(
                "no_rules_found",
                correlation_id=correlation_id,
                pessoa_id=person.pessoa_id,
            )
            return self._indeterminado(person, "Nenhuma regra encontrada para esta empresa e setor.")

        prompt = _build_prompt(person, rules, epi_attributes or [])

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
        except (json.JSONDecodeError, ValueError):
            # LLM às vezes gera JSON com aspas não escapadas na justificativa.
            # Tenta extrair status e justificativa via regex como fallback.
            data = _extract_fields_from_malformed_json(raw)
            if data is None:
                log.error(
                    "llm_call_failed",
                    correlation_id=correlation_id,
                    pessoa_id=person.pessoa_id,
                    error=f"JSON inválido e extração regex falhou: {raw[:80]}",
                )
                return self._indeterminado(person, f"Resposta do modelo não é JSON válido: {raw[:80]}")

        raw_status = data.get("status", "").strip()
        justificativa = data.get("justificativa", "").strip()

        # Normaliza variações de capitalização e conjugação que o LLM às vezes retorna
        # ex: "Não Conforme" → "Não conforme", "conforme" → "Conforme"
        _STATUS_ALIASES = {
            s.lower(): s for s in _VALID_STATUSES
        }
        _STATUS_ALIASES.update({
            "não conformo": "Não conforme",
            "não conforma": "Não conforme",
            "nao conforme": "Não conforme",
            "nao conformo": "Não conforme",
            "nao conforma": "Não conforme",
            "inconformidade": "Não conforme",
            "nao_conforme": "Não conforme",
            "não_conforme": "Não conforme",
        })
        status = _STATUS_ALIASES.get(raw_status.lower(), raw_status)

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


def _build_prompt(
    person: PersonDetection,
    rules: List[Rule],
    epi_attributes: List[EPIAttribute],
) -> str:
    if epi_attributes:
        attribute_lines = "\n".join(
            f"- {epi.label_pt}: "
            f"{'presente' if person.attributes.get(epi.name) is True else 'ausente' if person.attributes.get(epi.name) is False else 'incerto'}"
            for epi in epi_attributes
        )
    else:
        # Fallback genérico quando nenhum EPIAttribute foi configurado
        attribute_lines = "\n".join(
            f"- {k}: {'presente' if v is True else 'ausente' if v is False else 'incerto'}"
            for k, v in person.attributes.items()
        )
    rule_lines = "\n".join(f"- {r.rule} (fonte: {r.source})" for r in rules)
    return _PROMPT_TEMPLATE.format(attributes=attribute_lines, rules=rule_lines)


def _extract_fields_from_malformed_json(raw: str) -> dict | None:
    """
    Fallback para quando o LLM gera JSON com aspas não escapadas.
    Extrai status e justificativa via regex simples.
    Retorna None se não conseguir extrair ambos os campos.
    """
    status_match = re.search(r'"status"\s*:\s*"([^"]+)"', raw)
    # justificativa pode conter aspas — pega tudo entre o primeiro ": e o fim do objeto
    justificativa_match = re.search(r'"justificativa"\s*:\s*"(.+?)(?:"\s*\}|"\s*$)', raw, re.DOTALL)

    if not status_match or not justificativa_match:
        return None

    return {
        "status": status_match.group(1).strip(),
        "justificativa": justificativa_match.group(1).strip(),
    }
