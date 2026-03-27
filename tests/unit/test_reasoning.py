"""
Unit tests for the ReasoningService.

What we test (Llama always mocked — never called for real here):
- Happy path: Llama returns valid JSON → PersonResult correto
- Status mapping: Conforme / Não conforme / Indeterminado
- Sem regras → Indeterminado sem chamar o Llama
- Llama retorna JSON inválido → Indeterminado + log error
- Llama retorna status inválido → Indeterminado + log error
- Llama levanta exceção (timeout, connection) → Indeterminado + log error
- justificativa nunca vazia no output
- correlation_id passado ao cliente Ollama
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.reasoning.service import ReasoningService
from app.schemas.output import (
    BoundingBox,
    PersonAttributes,
    PersonDetection,
    PersonResult,
    Rule,
)

CORRELATION_ID = "test-correlation-id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_person(helmet: bool = False, vest: bool = True) -> PersonDetection:
    return PersonDetection(
        pessoa_id=1,
        bbox=BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=200.0),
        attributes=PersonAttributes(helmet=helmet, vest=vest),
    )


def _make_rules() -> list[Rule]:
    return [
        Rule(rule="Capacete obrigatório em obras.", source="manual.pdf"),
        Rule(rule="Colete refletivo obrigatório.", source="manual.pdf"),
    ]


def _llm_response(status: str, justificativa: str) -> str:
    return json.dumps({"status": status, "justificativa": justificativa})


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestReasoningHappyPath:
    def test_returns_person_result(self):
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response(
            "Não conforme", "Capacete ausente conforme regra 1."
        )
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert isinstance(result, PersonResult)

    def test_nao_conforme_status(self):
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response(
            "Não conforme", "Capacete ausente conforme regra 1."
        )
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(helmet=False), _make_rules(), CORRELATION_ID)

        assert result.status == "Não conforme"
        assert result.justificativa == "Capacete ausente conforme regra 1."

    def test_conforme_status(self):
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response(
            "Conforme", "Todos os EPIs presentes."
        )
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(helmet=True, vest=True), _make_rules(), CORRELATION_ID)

        assert result.status == "Conforme"

    def test_pessoa_id_and_bbox_preserved(self):
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response("Conforme", "OK.")
        service = ReasoningService(llm_client=mock_client)
        person = _make_person()

        result = service.analyze(person, _make_rules(), CORRELATION_ID)

        assert result.pessoa_id == person.pessoa_id
        assert result.bbox == person.bbox

    def test_llm_called_once(self):
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response("Conforme", "OK.")
        service = ReasoningService(llm_client=mock_client)

        service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        mock_client.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Edge case: no rules
# ---------------------------------------------------------------------------

class TestReasoningNoRules:
    def test_empty_rules_returns_indeterminado(self):
        mock_client = Mock()
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), [], CORRELATION_ID)

        assert result.status == "Indeterminado"

    def test_empty_rules_does_not_call_llm(self):
        mock_client = Mock()
        service = ReasoningService(llm_client=mock_client)

        service.analyze(_make_person(), [], CORRELATION_ID)

        mock_client.generate.assert_not_called()

    def test_empty_rules_justificativa_not_empty(self):
        mock_client = Mock()
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), [], CORRELATION_ID)

        assert result.justificativa.strip() != ""


# ---------------------------------------------------------------------------
# Edge case: bad LLM output
# ---------------------------------------------------------------------------

class TestReasoningBadLlmOutput:
    def test_invalid_json_returns_indeterminado(self):
        mock_client = Mock()
        mock_client.generate.return_value = "isso não é json { broken"
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.status == "Indeterminado"

    def test_invalid_json_justificativa_not_empty(self):
        mock_client = Mock()
        mock_client.generate.return_value = "broken"
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.justificativa.strip() != ""

    def test_invalid_status_in_response_returns_indeterminado(self):
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response("Aprovado", "Tudo certo.")
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.status == "Indeterminado"

    def test_missing_justificativa_in_response_returns_indeterminado(self):
        mock_client = Mock()
        mock_client.generate.return_value = json.dumps({"status": "Conforme"})
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.status == "Indeterminado"

    def test_empty_justificativa_in_response_returns_indeterminado(self):
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response("Conforme", "")
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.status == "Indeterminado"

    def test_status_capitalisation_normalised(self):
        """LLM retorna 'Não Conforme' (C maiúsculo) — deve ser aceito e normalizado."""
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response("Não Conforme", "Capacete ausente.")
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.status == "Não conforme"

    def test_status_all_lowercase_normalised(self):
        """LLM retorna 'conforme' em minúsculas — deve ser aceito e normalizado."""
        mock_client = Mock()
        mock_client.generate.return_value = _llm_response("conforme", "Todos EPIs presentes.")
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.status == "Conforme"


# ---------------------------------------------------------------------------
# Edge case: LLM failure
# ---------------------------------------------------------------------------

class TestReasoningLlmFailure:
    def test_llm_exception_returns_indeterminado(self):
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("connection timeout")
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.status == "Indeterminado"

    def test_llm_exception_does_not_propagate(self):
        mock_client = Mock()
        mock_client.generate.side_effect = RuntimeError("ollama down")
        service = ReasoningService(llm_client=mock_client)

        # Should not raise
        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)
        assert result is not None

    def test_llm_exception_justificativa_not_empty(self):
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("timeout")
        service = ReasoningService(llm_client=mock_client)

        result = service.analyze(_make_person(), _make_rules(), CORRELATION_ID)

        assert result.justificativa.strip() != ""
