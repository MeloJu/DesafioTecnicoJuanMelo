"""
Gera pares (imagem, texto) para fine-tuning do CLIP a partir do dataset público
keremberke/hard-hat-detection (HuggingFace Datasets).

Saída: data/clip_finetune/
    images/        → crops JPG
    labels.jsonl   → {"image": "...", "text": "...", "label": 1|-1}

Uso:
    python scripts/generate_clip_dataset.py [--split train] [--out data/clip_finetune]
"""
import argparse
import json
import pathlib

from datasets import load_dataset
from PIL import Image

DATASET = "keremberke/hard-hat-detection"
POSITIVE_TEXT = "a person wearing a hard hat"
NEGATIVE_TEXT = "a person not wearing a hard hat"


def _has_helmet(objects: dict) -> bool:
    """Verifica se algum objeto anotado é capacete."""
    categories = objects.get("category", [])
    return any("helmet" in str(c).lower() or "hardhat" in str(c).lower() for c in categories)


def main(split: str, out_dir: pathlib.Path) -> None:
    print(f"Baixando {DATASET} (split={split})...")
    ds = load_dataset(DATASET, split=split, trust_remote_code=True)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0
    for i, example in enumerate(ds):
        image = example.get("image")
        if image is None:
            skipped += 1
            continue

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        objects = example.get("objects", {})
        has_helmet = _has_helmet(objects)

        crop_path = images_dir / f"{i:05d}.jpg"
        image.save(crop_path, quality=90)

        records.append({
            "image": str(crop_path),
            "text": POSITIVE_TEXT if has_helmet else NEGATIVE_TEXT,
            "label": 1 if has_helmet else -1,
        })

    labels_path = out_dir / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    positives = sum(1 for r in records if r["label"] == 1)
    negatives = len(records) - positives
    print(f"Dataset gerado: {len(records)} exemplos ({positives} positivos, {negatives} negativos)")
    print(f"  Imagens: {images_dir}")
    print(f"  Labels:  {labels_path}")
    if skipped:
        print(f"  Ignorados (sem imagem): {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset CLIP para EPIs")
    parser.add_argument("--split", default="train", help="Split do HuggingFace (default: train)")
    parser.add_argument("--out", default="data/clip_finetune", help="Diretório de saída")
    args = parser.parse_args()
    main(split=args.split, out_dir=pathlib.Path(args.out))
