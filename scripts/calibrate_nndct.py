import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, "/project/scripts")

from recover_from_onnx import RecoveredDSCNN
sys.path.insert(0, "/project")
from pipeline.preprocessing import extract_log_mel_from_file
from pytorch_nndct.apis import torch_quantizer


MODEL_PATH = "/project/models/fp32/dscnn_medium_recovered.pth"
DATA_DIR = Path("/project/data/synthetic")
OUTPUT_DIR = "/project/models/nndct_calibrated"

CLASS_NAMES = [
    "down", "go", "left", "no", "off", "on",
    "right", "silence", "stop", "unknown", "up", "yes"
]


def collect_calibration_files():
    files = []

    for class_name in CLASS_NAMES:
        class_dir = DATA_DIR / class_name

        class_files = sorted(class_dir.glob("train_*.wav"))

        print(
            f"{class_name:8s}: "
            f"{len(class_files)} calibration files"
        )

        files.extend(class_files)

    return files


def main():

    print("Loading recovered model...")

    model = RecoveredDSCNN()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )
    model.eval()

    calibration_files = collect_calibration_files()

    print()
    print("Total calibration files:", len(calibration_files))

    if len(calibration_files) != 120:
        raise RuntimeError(
            f"Expected 120 calibration files, "
            f"found {len(calibration_files)}"
        )

    print()
    print("Extracting calibration features...")

    calibration_data = []

    for i, wav_path in enumerate(calibration_files):

        feature = extract_log_mel_from_file(str(wav_path))

        if feature.shape != (40, 98):
            raise RuntimeError(
                f"Unexpected feature shape for {wav_path}: "
                f"{feature.shape}"
            )

        x = feature[np.newaxis, np.newaxis, :, :].astype(
            np.float32
        )

        calibration_data.append(x)

        if (i + 1) % 20 == 0:
            print(
                f"  Processed {i + 1}/"
                f"{len(calibration_files)}"
            )

    print()
    print("Calibration feature shape:")
    print(calibration_data[0].shape)

    print()
    print("Creating NNDCT quantizer...")

    dummy_input = torch.from_numpy(calibration_data[0])

    quantizer = torch_quantizer(
        quant_mode=1,
        module=model,
        input_args=(dummy_input,),
        output_dir=OUTPUT_DIR,
        device=torch.device("cpu"),
    )

    quant_model = quantizer.quant_model

    print()
    print("Running calibration...")

    with torch.no_grad():

        for i, x in enumerate(calibration_data):

            input_tensor = torch.from_numpy(x)

            quant_model(input_tensor)

            if (i + 1) % 20 == 0:
                print(
                    f"  Calibrated {i + 1}/"
                    f"{len(calibration_data)}"
                )

    print()
    print("Calibration complete.")

    print()
    print("Exporting quantization configuration...")

    quantizer.export_quant_config()

    print()
    print("Calibration finished successfully.")

    print("Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
