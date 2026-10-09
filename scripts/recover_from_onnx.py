import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


ONNX_PATH = Path("/project/models/onnx/dscnn_medium.onnx")
OUTPUT_PATH = Path("/project/models/fp32/dscnn_medium_recovered.pth")


class RecoveredDSCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem
        self.stem = nn.Conv2d(
            1, 172,
            kernel_size=(10, 4),
            stride=(2, 1),
            padding=(4, 1),
            bias=True
        )

        # Four depthwise-separable blocks
        self.blocks = nn.ModuleList()

        for _ in range(4):
            self.blocks.append(
                nn.Sequential(
                    # Depthwise
                    nn.Conv2d(
                        172, 172,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=172,
                        bias=True
                    ),
                    nn.ReLU(),

                    # Pointwise
                    nn.Conv2d(
                        172, 172,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    ),
                    nn.ReLU()
                )
            )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(172, 172)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(172, 12)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


def get_initializer(model, name):
    import onnx
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    raise KeyError(name)


def main():
    print("Loading ONNX model:")
    print(ONNX_PATH)

    model_onnx = onnx.load(str(ONNX_PATH))

    print("\nONNX initializers:")
    for init in model_onnx.graph.initializer:
        print(f"  {init.name}: {list(init.dims)}")

    model = RecoveredDSCNN()

    # Expected ONNX initializer order based on the inspected graph.
    conv_names = [
        ("onnx::Conv_92", "onnx::Conv_93"),   # stem
        ("onnx::Conv_95", "onnx::Conv_96"),   # block 1 DW
        ("onnx::Conv_98", "onnx::Conv_99"),   # block 1 PW
        ("onnx::Conv_101", "onnx::Conv_102"), # block 2 DW
        ("onnx::Conv_104", "onnx::Conv_105"), # block 2 PW
        ("onnx::Conv_107", "onnx::Conv_108"), # block 3 DW
        ("onnx::Conv_110", "onnx::Conv_111"), # block 3 PW
        ("onnx::Conv_113", "onnx::Conv_114"), # block 4 DW
        ("onnx::Conv_116", "onnx::Conv_117"), # block 4 PW
    ]

    def copy_conv(layer, weight_name, bias_name):
        weight = get_initializer(model_onnx, weight_name)
        bias = get_initializer(model_onnx, bias_name)

        assert tuple(weight.shape) == tuple(layer.weight.shape), \
            f"Weight mismatch: {weight.shape} vs {tuple(layer.weight.shape)}"

        assert tuple(bias.shape) == tuple(layer.bias.shape), \
            f"Bias mismatch: {bias.shape} vs {tuple(layer.bias.shape)}"

        layer.weight.data.copy_(torch.from_numpy(weight))
        layer.bias.data.copy_(torch.from_numpy(bias))

    # Stem
    copy_conv(
        model.stem,
        conv_names[0][0],
        conv_names[0][1]
    )

    # Four blocks
    idx = 1

    for block_idx in range(4):
        block = model.blocks[block_idx]

        depthwise = block[0]
        pointwise = block[2]

        copy_conv(
            depthwise,
            conv_names[idx][0],
            conv_names[idx][1]
        )
        idx += 1

        copy_conv(
            pointwise,
            conv_names[idx][0],
            conv_names[idx][1]
        )
        idx += 1

    # Fully connected layers
    fc1_weight = get_initializer(model_onnx, "fc1.weight")
    fc1_bias = get_initializer(model_onnx, "fc1.bias")

    fc2_weight = get_initializer(model_onnx, "fc2.weight")
    fc2_bias = get_initializer(model_onnx, "fc2.bias")

    model.fc1.weight.data.copy_(torch.from_numpy(fc1_weight))
    model.fc1.bias.data.copy_(torch.from_numpy(fc1_bias))

    model.fc2.weight.data.copy_(torch.from_numpy(fc2_weight))
    model.fc2.bias.data.copy_(torch.from_numpy(fc2_bias))

    model.eval()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)

    print("\nRecovered model saved to:")
    print(OUTPUT_PATH)

    print("\nModel:")
    print(model)


if __name__ == "__main__":
    main()
