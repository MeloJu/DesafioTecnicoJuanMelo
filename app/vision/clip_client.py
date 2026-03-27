"""
CLIPClient — wrapper real para classificação zero-shot via HuggingFace.

Carregado uma única vez no __init__; roda em CPU.
Injetado no AttributeExtractor via DI — nunca instanciado em testes unitários.

classify() recebe os textos CLIP diretamente (positivo e negativo) em vez de
uma chave de atributo hardcodada. Os textos são definidos em EPIAttribute.

model_path permite carregar um modelo fine-tunado (salvo via CLIPModel.save_pretrained).
Se None, usa o modelo base "openai/clip-vit-base-patch32".
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_BASE_MODEL = "openai/clip-vit-base-patch32"


class CLIPClient:
    def __init__(self, model_path: str | None = None):
        model_id = model_path or _BASE_MODEL
        self._model = CLIPModel.from_pretrained(model_id)
        self._processor = CLIPProcessor.from_pretrained(model_id)

    def classify(
        self,
        image_crop: Image.Image,
        positive_text: str,
        negative_text: str,
    ) -> float:
        """Return similarity score [0,1] for the positive label."""
        inputs = self._processor(
            text=[positive_text, negative_text],
            images=image_crop,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        return float(probs[0][0])
