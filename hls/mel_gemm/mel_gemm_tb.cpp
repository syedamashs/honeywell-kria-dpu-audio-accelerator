#include <iostream>
#include <cmath>
#include "mel_gemm.h"

// Reference software matrix multiplication for testbench verification
void sw_mel_gemm_ref(
    float power_in[NUM_FRAMES][N_BINS],
    float mel_out[NUM_FRAMES][N_MELS]
) {
    for (int t = 0; t < NUM_FRAMES; t++) {
        for (int m = 0; m < N_MELS; m++) {
            float sum = 0.0f;
            for (int k = 0; k < N_BINS; k++) {
                sum += (float)MEL_WEIGHTS_ROM[m][k] * power_in[t][k];
            }
            mel_out[t][m] = sum;
        }
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " HLS C-SIMULATION TESTBENCH: Mel Filterbank GEMM Kernel \n";
    std::cout << "========================================================\n";

    static float power_test[NUM_FRAMES][N_BINS];
    static float expected_out[NUM_FRAMES][N_MELS];

    // Generate deterministic test pattern
    for (int t = 0; t < NUM_FRAMES; t++) {
        for (int k = 0; k < N_BINS; k++) {
            // Emulate a peak around bin 14 (440 Hz) plus background
            float peak = std::exp(-0.5f * std::pow((k - 14.0f) / 2.0f, 2.0f)) * 5.0f;
            power_test[t][k] = peak + 0.1f * ((t + k) % 10) / 10.0f;
        }
    }

    // Compute software reference
    sw_mel_gemm_ref(power_test, expected_out);

    // Prepare AXI-Stream inputs for DUT
    hls::stream<axis_pkt_t> power_stream("power_in_stream");
    hls::stream<axis_pkt_t> mel_stream("mel_out_stream");

    for (int t = 0; t < NUM_FRAMES; t++) {
        for (int k = 0; k < N_BINS; k++) {
            axis_pkt_t pkt;
            data_t val = (data_t)power_test[t][k];
            pkt.data.range(15, 0) = val.range(15, 0);
            pkt.last = (t == NUM_FRAMES - 1 && k == N_BINS - 1) ? 1 : 0;
            power_stream.write(pkt);
        }
    }

    // Call hardware kernel (DUT)
    std::cout << "[*] Executing mel_gemm_top DUT for " << NUM_FRAMES << " frames...\n";
    mel_gemm_top(power_stream, mel_stream, NUM_FRAMES);

    // Verify output stream against software reference
    float max_err = 0.0f;
    int mismatch_count = 0;

    for (int t = 0; t < NUM_FRAMES; t++) {
        for (int m = 0; m < N_MELS; m++) {
            if (mel_stream.empty()) {
                std::cerr << "[!] ERROR: Output stream ended prematurely at t=" << t << ", m=" << m << "\n";
                return 1;
            }
            axis_pkt_t pkt = mel_stream.read();
            out_t hw_val;
            hw_val.range(15, 0) = pkt.data.range(15, 0);
            float hw_f = (float)hw_val;
            float sw_f = expected_out[t][m];

            float err = std::abs(hw_f - sw_f);
            if (err > max_err) {
                max_err = err;
            }
            // Tolerance: 0.05 absolute error for 16-bit fixed point quantization
            if (err > 0.05f) {
                mismatch_count++;
                if (mismatch_count <= 5) {
                    std::cerr << "  Mismatch at [" << t << "][" << m << "]: HW=" << hw_f << " vs SW=" << sw_f << " (err=" << err << ")\n";
                }
            }
        }
    }

    std::cout << "[*] Verification results:\n";
    std::cout << "    Total outputs checked: " << NUM_FRAMES * N_MELS << "\n";
    std::cout << "    Max Absolute Error   : " << max_err << "\n";
    std::cout << "    Mismatches (>0.05)   : " << mismatch_count << "\n";

    if (mismatch_count == 0) {
        std::cout << "[OK] C-Simulation Testbench PASSED! Output matches reference.\n";
        return 0;
    } else {
        std::cerr << "[!] C-Simulation Testbench FAILED!\n";
        return 1;
    }
}
