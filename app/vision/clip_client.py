"""
CLIPClient — wrapper real para classificação zero-shot via HuggingFace.

Carregado uma única vez no __init__; roda em CPU.
Injetado no AttributeExtractor via DI — nunca instanciado em testes unitários.
"""
from typing import Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_TEXT_LABELS: dict[str, Tuple[str, str]] = {
    "helmet": (
        "a person wearing a hard hat",
        "a person not wearing a hard hat",
    ),
    "vest": (
        "a person wearing a high visibility vest",
        "a person not wearing a high visibility vest",
    ),
    "safety_boots": (
        "a person wearing safety boots",
        "a person wearing regular shoes or sandals",
    ),
    "gloves": (
        "a person wearing protective gloves",
        "a person not wearing gloves",
    ),
}


class CLIPClient:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)

    def classify(self, image_crop: Image.Image, attr: str) -> float:
        """Return similarity score [0,1] for the positive label of attr."""
        positive_text, negative_text = CLIP_TEXT_LABELS[attr]
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
