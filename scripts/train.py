"""
scripts/train.py
-----------------
Step 5 — Train DS-CNN on Google Speech Commands v2.

Usage:
    python scripts/train.py --variant medium --epochs 30 --batch_size 64

Outputs:
    models/fp32/dscnn_{variant}_best.pth        (best val accuracy)
    models/fp32/dscnn_{variant}_final.pth       (last epoch)
    models/onnx/dscnn_{variant}.onnx            (exported for DPU compile)
    results/raw/training_{variant}.csv          (epoch-level log)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.model     import build_model, count_params, count_macs
from pipeline.preprocessing import extract_log_mel_from_file, load_wav, extract_log_mel
from pipeline.utils     import (
    LABEL2IDX, N_MELS, NUM_FRAMES, NUM_CLASSES,
    set_seed, GLOBAL_SEED, save_versions, get_versions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KWSDataset(Dataset):
    """
    Keyword Spotting Dataset loader.

    Reads split .txt files written by download_dataset.py.
    Each line: /abs/path/to/file.wav,label,class_idx
    """

    def __init__(self, split_file: Path, augment: bool = False):
        self.entries  = []
        self.augment  = augment
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(",", 2)
                path, label, idx = parts[0], parts[1], int(parts[2])
                self.entries.append((path, label, idx))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        path, label, class_idx = self.entries[i]
        wav = load_wav(path)

        # Simple augmentation: SpecAugment-style time/freq masking
        if self.augment:
            wav = self._augment_wav(wav)

        feat = extract_log_mel(wav)   # (N_MELS, T) = (40, 101)

        # Add channel dim: (1, N_MELS, T) for Conv2D
        x = torch.from_numpy(feat).unsqueeze(0)   # (1, 40, 101)
        y = torch.tensor(class_idx, dtype=torch.long)
        return x, y

    def _augment_wav(self, wav: np.ndarray) -> np.ndarray:
        """Light waveform augmentation: random gain + small time shift."""
        wav = wav * np.random.uniform(0.8, 1.2)
        shift = np.random.randint(-1600, 1600)  # ±100ms
        wav = np.roll(wav, shift)
        if shift > 0:
            wav[:shift] = 0.0
        else:
            wav[shift:] = 0.0
        return wav.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scheduler  = None,
) -> tuple[float, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   DataLoader,
    criterion: nn.Module,
    device:   torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, out_path: Path, opset: int = 13) -> None:
    """
    Export model to ONNX (opset 13, static shape).

    DPU compile requirements:
    - Static input shape (no dynamic axes for DPU)
    - Opset 11 or 13 recommended for DPUCZDX8G
    - No dynamic control flow
    """
    model.eval()
    dummy = torch.zeros(1, 1, N_MELS, NUM_FRAMES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,   # static shape — required for DPU
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    print(f"[OK] ONNX exported -> {out_path}  (opset {opset})")

    # Verify ONNX graph
    try:
        import onnx
        m = onnx.load(str(out_path))
        onnx.checker.check_model(m)
        print(f"    ONNX check: OK  (inputs: {[i.name for i in m.graph.input]})")
    except ImportError:
        print("    [!] onnx not installed, skipping check")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",     default="medium",
                        choices=["small", "medium", "large", "gru"])
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed",        type=int, default=GLOBAL_SEED)
    parser.add_argument("--splits_dir",  default="data/processed/splits")
    parser.add_argument("--no_export_onnx", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    root       = Path(__file__).parent.parent
    splits_dir = root / args.splits_dir
    models_dir = root / "models"
    results_dir = root / "results" / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device   : {device}")
    print(f"[i] Variant  : {args.variant}")
    print(f"[i] Epochs   : {args.epochs}")

    # Save versions
    save_versions(results_dir / f"versions_{args.variant}.txt")

    # Datasets
    train_ds = KWSDataset(splits_dir / "train.txt", augment=True)
    val_ds   = KWSDataset(splits_dir / "val.txt",   augment=False)
    test_ds  = KWSDataset(splits_dir / "test.txt",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # Model
    model = build_model(args.variant).to(device)
    params = count_params(model)
    macs   = count_macs(model)
    print(f"[i] Params   : {params:,}")
    print(f"[i] MACs/inf : {macs/1e6:.1f} M")

    # Optimiser + scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    # CSV log
    csv_path = results_dir / f"training_{args.variant}.csv"
    csv_fh   = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_fh, fieldnames=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "epoch_time_s", "lr",
    ])
    writer.writeheader()

    best_val_acc = 0.0
    best_path    = models_dir / "fp32" / f"dscnn_{args.variant}_best.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>5} {'Tr Loss':>9} {'Tr Acc':>8} "
          f"{'Va Loss':>9} {'Va Acc':>8} {'Time':>7}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {tr_loss:>9.4f}  {tr_acc*100:>7.2f}%  "
              f"{va_loss:>9.4f}  {va_acc*100:>7.2f}%  {elapsed:>6.1f}s")

        writer.writerow({
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "train_acc": round(tr_acc, 6), "val_loss": round(va_loss, 6),
            "val_acc": round(va_acc, 6), "epoch_time_s": round(elapsed, 2),
            "lr": round(lr_now, 8),
        })
        csv_fh.flush()

        if va_acc >= best_val_acc or not best_path.exists():
            best_val_acc = va_acc
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "val_acc": va_acc, "params": params, "macs": macs,
                "args": vars(args), "versions": get_versions(),
            }, best_path)

    csv_fh.close()

    # Final checkpoint
    final_path = models_dir / "fp32" / f"dscnn_{args.variant}_final.pth"
    torch.save({"epoch": args.epochs, "state_dict": model.state_dict(),
                "val_acc": va_acc, "args": vars(args)}, final_path)

    # Test accuracy
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[OK] Test accuracy : {te_acc*100:.2f}%")
    print(f"[OK] Best val acc  : {best_val_acc*100:.2f}%")
    print(f"[OK] Checkpoints   : {best_path}")

    # ONNX export
    if not args.no_export_onnx:
        # Load best weights for ONNX export
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.cpu()
        onnx_path = models_dir / "onnx" / f"dscnn_{args.variant}.onnx"
        export_onnx(model, onnx_path)


if __name__ == "__main__":
    main()
