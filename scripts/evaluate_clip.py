"""
Avalia o CLIP (base vs. fine-tunado) no dataset sintético de EPIs.

Calcula accuracy, precision, recall e F1 para classificação binária
(tem capacete / não tem capacete) usando os labels conhecidos do dataset.

Uso:
    python scripts/evaluate_clip.py
    python scripts/evaluate_clip.py --data data/clip_finetune/test/labels.jsonl
    python scripts/evaluate_clip.py --finetuned models/clip_ppe
"""
import argparse
import json
import pathlib

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

POSITIVE_TEXT = "a person wearing a hard hat"
NEGATIVE_TEXT  = "a person not wearing a hard hat"
BASE_MODEL     = "openai/clip-vit-base-patch32"


def load_dataset(jsonl_path: str) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def predict(model: CLIPModel, processor: CLIPProcessor, image_path: str, threshold: float) -> int:
    """Retorna 1 (tem EPI) ou 0 (nao tem), dado um threshold de score."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[POSITIVE_TEXT, NEGATIVE_TEXT],
        images=image,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    score = float(probs[0][0])
    return 1 if score >= threshold else 0


def evaluate(model: CLIPModel, processor: CLIPProcessor, records: list[dict], threshold: float) -> dict:
    y_true, y_pred = [], []
    for r in records:
        true_label = 1 if r["label"] == 1 else 0
        pred_label = predict(model, processor, r["image"], threshold)
        y_true.append(true_label)
        y_pred.append(pred_label)

    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

    accuracy  = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy":  round(accuracy  * 100, 1),
        "precision": round(precision * 100, 1),
        "recall":    round(recall    * 100, 1),
        "f1":        round(f1        * 100, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": len(y_true),
    }


def load_model(model_path: str) -> tuple:
    print(f"Carregando: {model_path}")
    model     = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def print_results(name: str, metrics: dict) -> None:
    print(f"\n  {name}")
    print(f"  {'Accuracy:':<12} {metrics['accuracy']}%")
    print(f"  {'Precision:':<12} {metrics['precision']}%")
    print(f"  {'Recall:':<12} {metrics['recall']}%")
    print(f"  {'F1-score:':<12} {metrics['f1']}%")
    print(f"  Matriz de confusao: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia CLIP base vs. fine-tunado")
    parser.add_argument("--data",      default="data/clip_finetune/test/labels.jsonl")
    parser.add_argument("--finetuned", default="models/clip_ppe")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Score minimo para classificar como positivo (default: 0.50)")
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    if not data_path.exists():
        print(f"Dataset nao encontrado: {data_path}")
        print("Execute primeiro: python scripts/generate_clip_dataset.py --synthetic")
        raise SystemExit(1)

    records = load_dataset(args.data)
    print(f"Dataset: {len(records)} exemplos  "
          f"({sum(1 for r in records if r['label']==1)} positivos, "
          f"{sum(1 for r in records if r['label']!=1)} negativos)")
    print(f"Threshold: {args.threshold}")

    sep = "=" * 52
    print(f"\n{sep}")
    print(" CLIP — Avaliacao de Classificacao de EPI")
    print(f"{sep}")

    # Modelo base
    base_model, base_proc = load_model(BASE_MODEL)
    base_metrics = evaluate(base_model, base_proc, records, args.threshold)
    print_results("CLIP Base (openai/clip-vit-base-patch32)", base_metrics)

    # Modelo fine-tunado
    ft_path = pathlib.Path(args.finetuned)
    if ft_path.exists():
        ft_model, ft_proc = load_model(str(ft_path))
        ft_metrics = evaluate(ft_model, ft_proc, records, args.threshold)
        print_results(f"CLIP Fine-tunado ({ft_path})", ft_metrics)

        # Comparacao
        print(f"\n{'-' * 52}")
        print("  Ganho do fine-tuning:")
        for k in ["accuracy", "precision", "recall", "f1"]:
            delta = ft_metrics[k] - base_metrics[k]
            sinal = "+" if delta >= 0 else ""
            print(f"  {k.capitalize()+':':<12} {sinal}{delta:.1f} pp")
    else:
        print(f"\n  [aviso] Modelo fine-tunado nao encontrado em {ft_path}")
        print("  Execute: python scripts/finetune_clip.py --epochs 2 --batch 4")

    print(f"\n{sep}")
    print("  Nota: avaliacao no test split (held-out — nunca visto durante treino).")
    print("  Para metricas reais, use imagens reais de EPIs anotadas.")
    print(sep)


if __name__ == "__main__":
    main()
