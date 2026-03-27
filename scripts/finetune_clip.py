"""
Fine-tuning contrastivo do CLIP (openai/clip-vit-base-patch32) em dados de EPI.

Treina no split train/, monitora val_loss no split val/ a cada época e
salva o melhor modelo (lowest val_loss) — evitando overfitting.

Uso:
    # Gerar os splits primeiro:
    python scripts/generate_clip_dataset.py

    # Fine-tunar (CPU — batch pequeno):
    python scripts/finetune_clip.py --epochs 3 --batch 4

    # Fine-tunar (GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/finetune_clip.py --epochs 5 --batch 16

Saida padrao: models/clip_ppe/  (compativel com CLIPModel.from_pretrained)
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
        return {k: v.squeeze(0) for k, v in inputs.items()}


def _collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds  = F.normalize(text_embeds,  dim=-1)
    logits  = (image_embeds @ text_embeds.T) / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def _train_epoch(model: CLIPModel, loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler, device: str) -> float:
    model.train()
    total = 0.0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = contrastive_loss(outputs.image_embeds, outputs.text_embeds)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if step % max(1, len(loader) // 5) == 0:
            print(f"    step {step}/{len(loader)} · loss={loss.item():.4f}")
    return total / len(loader)


def _eval_epoch(model: CLIPModel, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = contrastive_loss(outputs.image_embeds, outputs.text_embeds)
            total += loss.item()
    return total / len(loader)


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Carregando modelo base: {BASE_MODEL}")

    processor = CLIPProcessor.from_pretrained(BASE_MODEL)
    model     = CLIPModel.from_pretrained(BASE_MODEL).to(device)

    train_ds = PPEDataset(args.train, processor)
    val_ds   = PPEDataset(args.val,   processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=_collate)

    print(f"Train: {len(train_ds)} exemplos · Val: {len(val_ds)} exemplos")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss   = _eval_epoch(model, val_loader, device)
        print(f"  train_loss={train_loss:.4f} · val_loss={val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(out)
            processor.save_pretrained(out)
            print(f"  -> melhor modelo salvo")
        else:
            print()

    print(f"\nTreino concluido.")
    print(f"  Melhor val_loss: {best_val_loss:.4f}")
    print(f"  Modelo salvo em: {out}")
    print(f"  Para avaliar no test split (held-out):")
    print(f"    python scripts/evaluate_clip.py")
    print(f"  Para usar no pipeline:")
    print(f"    python scripts/run_pipeline.py --clip-model {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning contrastivo do CLIP para EPIs")
    parser.add_argument("--train",   default="data/clip_finetune/train/labels.jsonl")
    parser.add_argument("--val",     default="data/clip_finetune/val/labels.jsonl")
    parser.add_argument("--epochs",  type=int,   default=3)
    parser.add_argument("--lr",      type=float, default=1e-5)
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size (use 4 em CPU)")
    parser.add_argument("--wd",      type=float, default=0.01)
    parser.add_argument("--output",  default="models/clip_ppe")
    args = parser.parse_args()

    for label, path in [("Train", args.train), ("Val", args.val)]:
        if not pathlib.Path(path).exists():
            print(f"{label} nao encontrado: {path}")
            print("Execute primeiro: python scripts/generate_clip_dataset.py")
            raise SystemExit(1)

    train(args)
