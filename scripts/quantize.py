"""
scripts/quantize.py
-------------------
Step 9 — Post-Training Quantization (PTQ) for DPUCZDX8G using Vitis AI (vai_q_pytorch).

Intended to run inside the Vitis AI Docker container:
    docker run -v $(pwd):/workspace -w /workspace xilinx/vitis-ai-pytorch-cpu:latest \
        python scripts/quantize.py --variant medium --quant_mode calib

Workflow:
  1. quant_mode='calib':
     Runs forward pass on 100–500 calibration clips through quantizer.quant_model.
     Generates quantization scaling parameters in models/quantized/quant_info.json.
  2. quant_mode='test':
     Evaluates INT8 accuracy on validation/test set.
     With --deploy, exports the quantized XIR xmodel (models/quantized/dscnn_{variant}_int.xmodel).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.model import build_model
from pipeline.utils import GLOBAL_SEED, N_MELS, NUM_FRAMES, set_seed
from scripts.train import KWSDataset

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"


def quantize_model(
    variant: str = "medium",
    quant_mode: str = "calib",
    subset_len: int = 100,
    batch_size: int = 16,
    deploy: bool = False,
    output_dir: Path = MODELS_DIR / "quantized",
):
    set_seed(GLOBAL_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = build_model(variant).to(device)
    model.eval()

    # Load FP32 checkpoint
    ckpt_path = MODELS_DIR / "fp32" / f"dscnn_{variant}_best.pth"
    if not ckpt_path.exists():
        ckpt_path = MODELS_DIR / "fp32" / f"dscnn_{variant}_final.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[*] Loaded trained FP32 weights from {ckpt_path}")
    else:
        print(f"[!] Warning: No checkpoint found at {ckpt_path}. Using initialized weights.")

    dummy_input = torch.zeros(1, 1, N_MELS, NUM_FRAMES, dtype=torch.float32)

    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError:
        print("\n" + "!" * 80)
        print("[!] ERROR: 'pytorch_nndct' not found.")
        print("    This script must be executed inside the Vitis AI Docker container:")
        print("    docker run -it -v $(pwd):/workspace -w /workspace xilinx/vitis-ai-pytorch-cpu:latest bash")
        print("    conda activate vitis-ai-pytorch")
        print("    python scripts/quantize.py --variant " + variant)
        print("!" * 80 + "\n")
        return

    print(f"[*] Initializing Vitis AI quantizer (mode={quant_mode}, variant={variant})...")
    quantizer = torch_quantizer(
        quant_mode=quant_mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=str(output_dir),
        device=device,
    )
    quant_model = quantizer.quant_model

    # Load calibration / validation data
    split_file = SPLITS_DIR / ("train.txt" if quant_mode == "calib" else "test.txt")
    dataset = KWSDataset(split_file, augment=False)
    indices = list(range(min(subset_len, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    print(f"[*] Running {quant_mode} on {len(subset)} samples...")
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = quant_model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    acc = correct / max(total, 1)
    print(f"[OK] Quantized model ({quant_mode}) Accuracy: {acc * 100:.2f}% ({correct}/{total})")

    if quant_mode == "calib":
        quantizer.export_quant_config()
        print(f"[OK] Exported quantization calibration config -> {output_dir}")
    elif quant_mode == "test" and deploy:
        quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=True)
        print(f"[OK] Exported deployed XIR model -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Vitis AI Quantization (vai_q_pytorch)")
    parser.add_argument("--variant", default="medium", choices=["small", "medium", "large", "gru"])
    parser.add_argument("--quant_mode", default="calib", choices=["calib", "test"])
    parser.add_argument("--subset_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--deploy", action="store_true", help="Export xmodel when in test mode")
    parser.add_argument("--out_dir", default="models/quantized")
    args = parser.parse_args()

    quantize_model(
        variant=args.variant,
        quant_mode=args.quant_mode,
        subset_len=args.subset_len,
        batch_size=args.batch_size,
        deploy=args.deploy,
        output_dir=ROOT / args.out_dir,
    )


if __name__ == "__main__":
    main()
