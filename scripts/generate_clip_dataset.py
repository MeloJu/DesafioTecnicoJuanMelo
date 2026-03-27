"""
Gera pares (imagem, texto) para fine-tuning do CLIP em dados de EPI.

Tenta carregar o dataset público 'ShelterX/hardhat-detection' ou
'phiyodr/coco2017' (objetos "person"). Se nenhum estiver disponível ou
houver problema de rede, gera um dataset sintético local suficiente para
demonstrar o pipeline de treinamento.

Saída: data/clip_finetune/
    images/        → crops JPG (224x224)
    labels.jsonl   → {"image": "...", "text": "...", "label": 1|-1}

Uso:
    python scripts/generate_clip_dataset.py [--out data/clip_finetune] [--samples 200]
    python scripts/generate_clip_dataset.py --synthetic  # força dataset sintético
"""
import argparse
import json
import pathlib
import random

from PIL import Image, ImageDraw

POSITIVE_TEXT = "a person wearing a hard hat"
NEGATIVE_TEXT = "a person not wearing a hard hat"
IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Synthetic dataset (fallback — sempre funciona, sem internet)
# ---------------------------------------------------------------------------

def _make_synthetic_image(has_helmet: bool, seed: int) -> Image.Image:
    """
    Gera uma imagem sintética simples que representa visualmente se a pessoa
    tem ou não capacete. Suficiente para demonstrar o pipeline de fine-tuning.
    """
    rng = random.Random(seed)
    bg_color = (rng.randint(180, 230), rng.randint(180, 220), rng.randint(160, 210))
    img = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(img)

    # Corpo (retângulo)
    cx, cy = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    shirt_color = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
    draw.rectangle([cx - 30, cy, cx + 30, cy + 70], fill=shirt_color)

    # Cabeça (elipse)
    head_color = (rng.randint(180, 240), rng.randint(140, 190), rng.randint(120, 170))
    draw.ellipse([cx - 20, cy - 50, cx + 20, cy], fill=head_color)

    # Capacete (se tiver)
    if has_helmet:
        helmet_color = (rng.randint(200, 255), rng.randint(150, 200), 0)
        draw.ellipse([cx - 22, cy - 55, cx + 22, cy - 30], fill=helmet_color)
        draw.rectangle([cx - 24, cy - 36, cx + 24, cy - 30], fill=helmet_color)

    return img


def generate_synthetic(out_dir: pathlib.Path, n_samples: int) -> None:
    """Gera dataset sintético sem precisar de internet."""
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(n_samples):
        has_helmet = i % 2 == 0  # alterna positivo/negativo
        img = _make_synthetic_image(has_helmet, seed=i)
        crop_path = images_dir / f"{i:05d}.jpg"
        img.save(crop_path, quality=90)
        records.append({
            "image": str(crop_path),
            "text": POSITIVE_TEXT if has_helmet else NEGATIVE_TEXT,
            "label": 1 if has_helmet else -1,
        })

    _save_jsonl(records, out_dir)
    positives = n_samples // 2
    print(f"Dataset sintético gerado: {len(records)} exemplos ({positives} positivos, {positives} negativos)")
    print(f"  Imagens: {images_dir}")
    print(f"  Labels:  {out_dir / 'labels.jsonl'}")


# ---------------------------------------------------------------------------
# HuggingFace dataset (tentativa com fallback automático)
# ---------------------------------------------------------------------------

_HF_DATASETS = [
    # (dataset_id, config_or_None, split, label_field_fn)
    ("Andyrasika/hardhat-detection", None, "train", lambda ex: bool(ex.get("label", 0))),
    ("blueye/hard-hat-detection", None, "train", lambda ex: bool(ex.get("label", 0))),
]


def _try_hf_dataset(out_dir: pathlib.Path, n_samples: int) -> bool:
    """
    Tenta carregar um dataset HuggingFace compatível.
    Retorna True se conseguiu, False caso contrário.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return False

    for dataset_id, config, split, label_fn in _HF_DATASETS:
        try:
            print(f"Tentando carregar {dataset_id}...")
            kwargs = {"split": split}
            if config:
                kwargs["name"] = config
            ds = load_dataset(dataset_id, **kwargs)
            if n_samples and n_samples < len(ds):
                ds = ds.select(range(n_samples))

            images_dir = out_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            records = []

            for i, example in enumerate(ds):
                image = example.get("image")
                if image is None:
                    continue
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                image = image.convert("RGB").resize(IMAGE_SIZE)
                crop_path = images_dir / f"{i:05d}.jpg"
                image.save(crop_path, quality=90)
                has_helmet = label_fn(example)
                records.append({
                    "image": str(crop_path),
                    "text": POSITIVE_TEXT if has_helmet else NEGATIVE_TEXT,
                    "label": 1 if has_helmet else -1,
                })

            if records:
                _save_jsonl(records, out_dir)
                positives = sum(1 for r in records if r["label"] == 1)
                print(f"Dataset {dataset_id}: {len(records)} exemplos ({positives} positivos)")
                return True

        except Exception as exc:
            print(f"  Falhou ({type(exc).__name__}): {exc}")
            continue

    return False


def _save_jsonl(records: list, out_dir: pathlib.Path) -> None:
    labels_path = out_dir / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(out_dir: pathlib.Path, n_samples: int, force_synthetic: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force_synthetic:
        success = _try_hf_dataset(out_dir, n_samples)
        if success:
            return
        print("\nDatasets HuggingFace indisponíveis — usando dataset sintético local.")

    generate_synthetic(out_dir, n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset CLIP para fine-tuning em EPIs")
    parser.add_argument("--out", default="data/clip_finetune", help="Diretório de saída")
    parser.add_argument("--samples", type=int, default=200, help="Número máximo de exemplos (default: 200)")
    parser.add_argument("--synthetic", action="store_true", help="Força dataset sintético (sem internet)")
    args = parser.parse_args()
    main(out_dir=pathlib.Path(args.out), n_samples=args.samples, force_synthetic=args.synthetic)
