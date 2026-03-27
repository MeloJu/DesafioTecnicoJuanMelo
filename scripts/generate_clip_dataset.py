"""
Gera pares (imagem, texto) para fine-tuning do CLIP em dados de EPI,
com separação correta em train / val / test (sem data leakage).

Tenta carregar keremberke/protective-equipment-detection (HuggingFace),
que possui splits nativos train/valid/test. Se indisponível, gera um
dataset sintético local com 300 exemplos divididos 70/15/15.

Saída: data/clip_finetune/
    train/  labels.jsonl + images/   (70% — usado no finetune_clip.py)
    val/    labels.jsonl + images/   (15% — monitorado durante treino)
    test/   labels.jsonl + images/   (15% — NUNCA visto durante treino)

Uso:
    python scripts/generate_clip_dataset.py
    python scripts/generate_clip_dataset.py --synthetic        # força local
    python scripts/generate_clip_dataset.py --samples 600      # mais exemplos
"""
import argparse
import json
import pathlib
import random

from PIL import Image, ImageDraw

POSITIVE_TEXT = "a person wearing a hard hat"
NEGATIVE_TEXT = "a person not wearing a hard hat"
IMAGE_SIZE    = (224, 224)

HF_DATASET    = "keremberke/protective-equipment-detection"
HF_SPLIT_MAP  = {"train": "train", "val": "valid", "test": "test"}
POSITIVE_CATS = {"helmet"}
NEGATIVE_CATS = {"no_helmet"}


# ---------------------------------------------------------------------------
# Helpers comuns
# ---------------------------------------------------------------------------

def _save_jsonl(records: list, split_dir: pathlib.Path) -> None:
    labels_path = split_dir / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _split_records(records: list, out_dir: pathlib.Path,
                   train_frac: float = 0.70, val_frac: float = 0.15) -> None:
    """Divide records em train/val/test e salva cada split em subdiretório."""
    n       = len(records)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    splits  = {
        "train": records[:n_train],
        "val":   records[n_train : n_train + n_val],
        "test":  records[n_train + n_val:],
    }
    for name, recs in splits.items():
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        _save_jsonl(recs, split_dir)
        pos = sum(1 for r in recs if r["label"] == 1)
        print(f"  {name:5s}: {len(recs):>4} exemplos  ({pos} pos / {len(recs)-pos} neg)")


def _print_summary(out_dir: pathlib.Path) -> None:
    total = 0
    for split in ("train", "val", "test"):
        p = out_dir / split / "labels.jsonl"
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            total += n
    print(f"\nTotal: {total} exemplos -> {out_dir}")
    print("Splits sao disjuntos - sem data leakage.")


# ---------------------------------------------------------------------------
# Dataset sintetico (fallback — sempre funciona, sem internet)
# ---------------------------------------------------------------------------

def _make_synthetic_image(has_helmet: bool, seed: int) -> Image.Image:
    rng = random.Random(seed)
    bg  = (rng.randint(180, 230), rng.randint(180, 220), rng.randint(160, 210))
    img = Image.new("RGB", IMAGE_SIZE, bg)
    draw = ImageDraw.Draw(img)
    cx, cy = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    shirt = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
    draw.rectangle([cx - 30, cy, cx + 30, cy + 70], fill=shirt)
    head = (rng.randint(180, 240), rng.randint(140, 190), rng.randint(120, 170))
    draw.ellipse([cx - 20, cy - 50, cx + 20, cy], fill=head)
    if has_helmet:
        hcol = (rng.randint(200, 255), rng.randint(150, 200), 0)
        draw.ellipse([cx - 22, cy - 55, cx + 22, cy - 30], fill=hcol)
        draw.rectangle([cx - 24, cy - 36, cx + 24, cy - 30], fill=hcol)
    return img


def generate_synthetic(out_dir: pathlib.Path, n_samples: int) -> None:
    """Gera dataset sintetico e divide em train/val/test sem overlap."""
    print(f"Gerando dataset sintetico: {n_samples} exemplos...")
    images_dir = out_dir / "images_all"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(n_samples):
        has_helmet = (i % 2 == 0)
        img = _make_synthetic_image(has_helmet, seed=i)
        path = images_dir / f"{i:05d}.jpg"
        img.save(path, quality=90)
        records.append({
            "image": str(path),
            "text":  POSITIVE_TEXT if has_helmet else NEGATIVE_TEXT,
            "label": 1 if has_helmet else -1,
        })

    _split_records(records, out_dir)
    _print_summary(out_dir)


# ---------------------------------------------------------------------------
# HuggingFace dataset (com splits nativos)
# ---------------------------------------------------------------------------

def _extract_label(objects: dict) -> int | None:
    """
    Retorna 1 se a imagem contiver 'helmet', -1 se 'no_helmet', None se ambíguo.
    """
    cats = {str(c).lower() for c in objects.get("category", [])}
    has_positive = bool(cats & POSITIVE_CATS)
    has_negative = bool(cats & NEGATIVE_CATS)
    if has_positive and not has_negative:
        return 1
    if has_negative and not has_positive:
        return -1
    return None  # ambíguo — descartar


def _process_hf_split(ds, split_name: str, out_dir: pathlib.Path) -> list[dict]:
    split_dir  = out_dir / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, example in enumerate(ds):
        image = example.get("image")
        if image is None:
            continue
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB").resize(IMAGE_SIZE)

        label = _extract_label(example.get("objects", {}))
        if label is None:
            continue

        path = images_dir / f"{i:05d}.jpg"
        image.save(path, quality=90)
        records.append({
            "image": str(path),
            "text":  POSITIVE_TEXT if label == 1 else NEGATIVE_TEXT,
            "label": label,
        })

    _save_jsonl(records, split_dir)
    pos = sum(1 for r in records if r["label"] == 1)
    print(f"  {split_name:5s}: {len(records):>4} exemplos  ({pos} pos / {len(records)-pos} neg)")
    return records


def try_hf_dataset(out_dir: pathlib.Path, max_per_split: int) -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        return False

    try:
        print(f"Baixando {HF_DATASET}...")
        ds_all = load_dataset(HF_DATASET, name="full")
    except Exception as exc:
        print(f"  Falhou ({type(exc).__name__}): {exc}")
        return False

    total = 0
    for local_name, hf_name in HF_SPLIT_MAP.items():
        if hf_name not in ds_all:
            print(f"  Split '{hf_name}' nao encontrado — abortando HF.")
            return False
        ds = ds_all[hf_name]
        if max_per_split and max_per_split < len(ds):
            ds = ds.select(range(max_per_split))
        recs = _process_hf_split(ds, local_name, out_dir)
        total += len(recs)

    if total == 0:
        return False

    _print_summary(out_dir)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(out_dir: pathlib.Path, n_samples: int, force_synthetic: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force_synthetic:
        ok = try_hf_dataset(out_dir, max_per_split=n_samples // 3)
        if ok:
            return
        print("\nHuggingFace indisponivel — usando dataset sintetico local.")

    generate_synthetic(out_dir, n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera dataset CLIP com train/val/test split")
    parser.add_argument("--out",       default="data/clip_finetune")
    parser.add_argument("--samples",   type=int, default=300,
                        help="Total de exemplos sinteticos (default: 300 → 210/45/45)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Forca dataset sintetico local (sem internet)")
    args = parser.parse_args()
    main(pathlib.Path(args.out), args.samples, args.synthetic)
