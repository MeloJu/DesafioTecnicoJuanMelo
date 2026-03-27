"""
Fine-tuning contrastivo do CLIP (openai/clip-vit-base-patch32) em dados de EPI.

Technique: contrastive loss simétrica sobre a matriz de similaridade imagem×texto,
idêntica ao pré-treinamento original do CLIP.

Uso:
    # Gerar dataset primeiro:
    python scripts/generate_clip_dataset.py

    # Fine-tunar:
    python scripts/finetune_clip.py [--epochs 3] [--lr 1e-5] [--batch 16]

    # Com GPU (muito mais rápido):
    CUDA_VISIBLE_DEVICES=0 python scripts/finetune_clip.py --epochs 5

Saída padrão: models/clip_ppe/  (compatível com CLIPModel.from_pretrained)
"""
import argparse
import json
import pathlib

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

BASE_MODEL = "openai/clip-vit-base-patch32"


class PPEDataset(Dataset):
    """Pares (image_crop, texto) carregados de labels.jsonl."""

    def __init__(self, jsonl_path: str, processor: CLIPProcessor) -> None:
        with open(jsonl_path, encoding="utf-8") as f:
            self._records = [json.loads(line) for line in f if line.strip()]
        self._processor = processor

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        r = self._records[idx]
        image = Image.open(r["image"]).convert("RGB")
        inputs = self._processor(
            text=[r["text"]],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        # Remove a dimensão de batch adicionada pelo processor
        return {k: v.squeeze(0) for k, v in inputs.items()}


def _collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Cross-entropy simétrica sobre a matriz de similaridade coseno normalizada."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = (image_embeds @ text_embeds.T) / temperature
    targets = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return (loss_i + loss_t) / 2


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Carregando modelo base: {BASE_MODEL}")
    processor = CLIPProcessor.from_pretrained(BASE_MODEL)
    model = CLIPModel.from_pretrained(BASE_MODEL).to(device)

    dataset = PPEDataset(args.data, processor)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    print(f"Dataset: {len(dataset)} exemplos · {len(loader)} batches por época")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = contrastive_loss(outputs.image_embeds, outputs.text_embeds)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % max(1, len(loader) // 5) == 0:
                print(f"  Epoch {epoch}/{args.epochs} · step {step}/{len(loader)} · loss={loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} concluída · avg_loss={avg_loss:.4f}")

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"\nModelo salvo em: {out}")
    print("Para usar no pipeline:")
    print(f"  pipeline = create_pipeline(clip_model_path='{out}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning contrastivo do CLIP para EPIs")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16; use 4 em CPU)")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--data", default="data/clip_finetune/labels.jsonl", help="Path para labels.jsonl")
    parser.add_argument("--output", default="models/clip_ppe", help="Diretório de saída do modelo")
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    if not data_path.exists():
        print(f"Dataset não encontrado: {data_path}")
        print("Execute primeiro: python scripts/generate_clip_dataset.py")
        raise SystemExit(1)

    train(args)
