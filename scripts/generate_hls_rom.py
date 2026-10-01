"""
scripts/generate_hls_rom.py
---------------------------
Exports the exact (40, 257) Slaney mel filterbank matrix as C++ ROM table
for the Vitis HLS kernel: hls/mel_gemm/mel_weights_rom.cpp
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.preprocessing import build_mel_filterbank


def generate_hls_rom():
    M = build_mel_filterbank()
    out_file = ROOT / "hls" / "mel_gemm" / "mel_weights_rom.cpp"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        f.write('#include "mel_gemm.h"\n\n')
        f.write("// Auto-generated from pipeline.preprocessing.build_mel_filterbank()\n")
        f.write("const weight_t MEL_WEIGHTS_ROM[N_MELS][N_BINS] = {\n")
        for m in range(40):
            f.write("    {")
            row_str = ", ".join(f"{M[m, k]:.8f}" for k in range(257))
            f.write(row_str)
            f.write("},\n")
        f.write("};\n")

    print(f"[OK] HLS ROM generated -> {out_file} ({out_file.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_hls_rom()
