"""
OllamaLLM — wrapper real para Llama via Ollama.

Injetado no ReasoningService via DI — nunca instanciado em testes unitários.
Requer Ollama rodando localmente com o modelo configurado disponível.
"""
import ollama


class OllamaLLM:
    def __init__(self, model: str = "llama3.2"):
        self._model = model

    def generate(self, prompt: str, correlation_id: str = "") -> str:
        # correlation_id não é usado aqui — mantido para paridade com o contrato
        # do mock de testes (LLMMock) e para rastreamento futuro via logs.
        response = ollama.generate(model=self._model, prompt=prompt)
        return response["response"]
