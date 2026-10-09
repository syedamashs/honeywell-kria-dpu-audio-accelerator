import sys
from pathlib import Path

import torch

sys.path.insert(0, "/project/scripts")
sys.path.insert(0, "/project")

from recover_from_onnx import RecoveredDSCNN
from pytorch_nndct.apis import torch_quantizer


MODEL_PATH = "/project/models/fp32/dscnn_medium_recovered.pth"
QUANT_CONFIG = "/project/models/nndct_calibrated/quant_info.json"
OUTPUT_DIR = "/project/models/nndct_xmodel"

DEVICE = torch.device("cpu")


print("Loading recovered FP32 model...")

model = RecoveredDSCNN()
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

dummy_input = torch.randn(1, 1, 40, 98)

print("Model loaded.")
print("Input shape:", tuple(dummy_input.shape))
print("Quant config:", QUANT_CONFIG)
print("Output directory:", OUTPUT_DIR)

print("\nCreating NNDCT quantizer in test/deploy mode...")

quantizer = torch_quantizer(
    quant_mode=2,
    module=model,
    input_args=(dummy_input,),
    output_dir="/project/models/nndct_calibrated",
    device=DEVICE,
)

print("\nRunning quantized model once...")

quant_model = quantizer.quant_model

with torch.no_grad():
    output = quant_model(dummy_input)

if isinstance(output, tuple):
    output = output[0]

print("Output shape:", tuple(output.shape))
print("Output sample:", output.flatten()[:5])

print("\nExporting XMODEL...")

quantizer.export_xmodel(
    output_dir=OUTPUT_DIR,
    deploy_check=True,
)

print("\nXMODEL export finished.")

print("\nGenerated files:")
for p in sorted(Path(OUTPUT_DIR).glob("*")):
    print(" ", p)
