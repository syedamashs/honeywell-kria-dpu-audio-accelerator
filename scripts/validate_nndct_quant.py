import torch
import sys
from pathlib import Path

sys.path.insert(0, "/project/scripts")
from recover_from_onnx import RecoveredDSCNN

from pytorch_nndct.apis import torch_quantizer


MODEL_PATH = "/project/models/fp32/dscnn_medium_recovered.pth"
OUTPUT_DIR = "/project/models/nndct_test"

model = RecoveredDSCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 40, 98)

print("Creating NNDCT quantizer...")

quantizer = torch_quantizer(
    quant_mode=1,
    module=model,
    input_args=(dummy_input,),
    output_dir=OUTPUT_DIR,
    device=torch.device("cpu"),
)

print("\nNNDCT quantizer created successfully.")

quant_model = quantizer.quant_model

print("\nRunning quantized model once...")

with torch.no_grad():
    output = quant_model(dummy_input)

print("Quantized output shape:", output.shape)
print("Quantized output sample:", output[0, :5])

print("\nNNDCT quantization test PASSED.")
