import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/project")

from pipeline.preprocessing import extract_log_mel_from_file
from onnxruntime.quantization import CalibrationDataReader

import vai_q_onnx


PROJECT = Path("/project")
MODEL_PATH = PROJECT / "models/onnx/dscnn_medium.onnx"
OUTPUT_DIR = PROJECT / "models/quantized"

NUM_SAMPLES_PER_CLASS = 10


class KWSCalibrationDataReader(CalibrationDataReader):
    def __init__(self, wav_files):
        self.wav_files = list(wav_files)
        self.index = 0

    def get_next(self):
        if self.index >= len(self.wav_files):
            return None

        wav_path = self.wav_files[self.index]
        self.index += 1

        feature = extract_log_mel_from_file(wav_path)

        input_tensor = feature[np.newaxis, np.newaxis, :, :].astype(
            np.float32
        )

        return {"input": input_tensor}

    def rewind(self):
        self.index = 0


def collect_calibration_files():
    root = PROJECT / "data/synthetic"

    class_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir()
    )

    wav_files = []

    for class_dir in class_dirs:
        train_files = sorted(class_dir.glob("train_*.wav"))

        if len(train_files) < NUM_SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"{class_dir.name}: only {len(train_files)} "
                f"training files found"
            )

        wav_files.extend(train_files[:NUM_SAMPLES_PER_CLASS])

    return wav_files


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model:")
    print(" ", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    wav_files = collect_calibration_files()

    print(f"Calibration samples: {len(wav_files)}")

    reader = KWSCalibrationDataReader(wav_files)

    output_model = OUTPUT_DIR / "dscnn_medium_int8.onnx"

    print("\nStarting Vitis AI ONNX quantization...")
    print("Output:", output_model)

    vai_q_onnx.quantize_static(
        model_input=str(MODEL_PATH),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=vai_q_onnx.VitisQuantFormat.FixNeuron,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    )

    print("\nQuantization completed.")
    print("INT8 model:", output_model)


if __name__ == "__main__":
    main()
