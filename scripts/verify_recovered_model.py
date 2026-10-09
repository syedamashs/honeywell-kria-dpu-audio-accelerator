import torch
import onnxruntime as ort
import numpy as np
import sys

sys.path.insert(0, "/project/scripts")
from recover_from_onnx import RecoveredDSCNN

PTH = "/project/models/fp32/dscnn_medium_recovered.pth"
ONNX = "/project/models/onnx/dscnn_medium.onnx"

# Load recovered PyTorch model
model = RecoveredDSCNN()
model.load_state_dict(torch.load(PTH, map_location="cpu"))
model.eval()

# ONNX Runtime
session = ort.InferenceSession(
    ONNX,
    providers=["CPUExecutionProvider"]
)

# Fixed random input for reproducibility
np.random.seed(42)
x = np.random.randn(1, 1, 40, 98).astype(np.float32)

# PyTorch
with torch.no_grad():
    torch_output = model(torch.from_numpy(x)).numpy()

# ONNX
onnx_output = session.run(
    ["logits"],
    {"input": x}
)[0]

# Compare
diff = np.abs(torch_output - onnx_output)

print("Input shape:       ", x.shape)
print("PyTorch output:    ", torch_output.shape)
print("ONNX output:       ", onnx_output.shape)

print("\nMaximum absolute difference:")
print(diff.max())

print("\nMean absolute difference:")
print(diff.mean())

print("\nPyTorch logits:")
print(torch_output)

print("\nONNX logits:")
print(onnx_output)

if diff.max() < 1e-4:
    print("\nPASS: Recovered PyTorch model matches ONNX.")
else:
    print("\nFAIL: Recovered PyTorch model does NOT match ONNX.")
    sys.exit(1)
