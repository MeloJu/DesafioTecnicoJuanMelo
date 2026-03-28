"""
EPIAttribute — configuração de um atributo de EPI (Equipamento de Proteção Individual).

Centraliza em um único dataclass tudo que o pipeline precisa saber sobre um EPI:
  - name: chave interna usada como chave no dict de atributos (ex: "helmet")
  - label_pt: nome em português para RAG query e prompt do LLM (ex: "capacete")
  - clip_positive: texto CLIP para presença do EPI
  - clip_negative: texto CLIP para ausência do EPI
  - threshold_positive: score mínimo para confirmar presença (True)
  - threshold_negative: score máximo para confirmar ausência (False)
  - scores entre os dois thresholds → None (zona de incerteza → Indeterminado)

Para adicionar um novo EPI: crie um EPIAttribute e adicione à lista relevante.
Nenhum outro arquivo precisa ser alterado.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class EPIAttribute:
    name: str                         # chave interna: "helmet", "safety_glasses"
    label_pt: str                     # nome PT para RAG e prompt: "capacete"
    clip_positive: str                # texto CLIP positivo (presença)
    clip_negative: str                # texto CLIP negativo (ausência)
    threshold_positive: float = 0.60  # score >= threshold_positive → True
    threshold_negative: float = 0.40  # score <= threshold_negative → False


DEFAULT_EPI_ATTRIBUTES: List[EPIAttribute] = [
    # Threshold mais alto: borda circular do capacete é visualmente distinta,
    # o que torna o CLIP mais confiante — reduz falsos positivos.
    EPIAttribute(
        name="helmet",
        label_pt="capacete",
        clip_positive="a person wearing a hard hat",
        clip_negative="a person not wearing a hard hat",
        threshold_positive=0.65,
        threshold_negative=0.35,
    ),
    # Threshold padrão: colete tem alta variação de cor e formato entre empresas.
    EPIAttribute(
        name="vest",
        label_pt="colete refletivo",
        clip_positive="a person wearing a high visibility vest",
        clip_negative="a person not wearing a high visibility vest",
        threshold_positive=0.60,
        threshold_negative=0.40,
    ),
    # Zona de incerteza mais ampla (0.30–0.70): botas sofrem oclusão frequente
    # por objetos no chão, paletes e ângulos de câmera baixos.
    EPIAttribute(
        name="safety_boots",
        label_pt="botas de segurança",
        clip_positive="a person wearing safety boots",
        clip_negative="a person wearing regular shoes or sandals",
        threshold_positive=0.70,
        threshold_negative=0.30,
    ),
    # Threshold padrão: luvas têm variação de cor e tamanho, mesma incerteza do colete.
    EPIAttribute(
        name="gloves",
        label_pt="luvas de proteção",
        clip_positive="a person wearing protective gloves",
        clip_negative="a person not wearing gloves",
        threshold_positive=0.60,
        threshold_negative=0.40,
    ),
]
