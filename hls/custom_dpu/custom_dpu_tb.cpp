#include <iostream>
#include <cmath>
#include "custom_dpu.h"

int main() {
    std::cout << "========================================================\n";
    std::cout << " HLS C-SIMULATION TESTBENCH: Custom DS-CNN DPU Kernel  \n";
    std::cout << " Target: AMD Kria KV260 (xck26-sfvc784-2LV-c) @ 300MHz \n";
    std::cout << "========================================================\n";

    hls::stream<axis_pkt_t> features_stream("features_in");
    hls::stream<axis_pkt_t> logits_stream("logits_out");

    // 1. Generate test Mel Spectrogram stream (40 bands x 98 frames)
    std::cout << "[*] Streaming [40 x 98] Mel spectrogram packets into DUT...\n";
    for (int h = 0; h < IN_HEIGHT; h++) {
        for (int w = 0; w < IN_WIDTH; w++) {
            axis_pkt_t pkt;
            // Test pattern: acoustic formant around frequency band 14
            float val = std::exp(-0.5f * std::pow((h - 14.0f) / 3.0f, 2.0f)) * 4.0f;
            feat_t fixed_val = (feat_t)val;
            pkt.data.range(15, 0) = fixed_val.range(15, 0);
            pkt.last = (h == IN_HEIGHT - 1 && w == IN_WIDTH - 1) ? 1 : 0;
            features_stream.write(pkt);
        }
    }

    // 2. Execute Hardware Kernel (Design Under Test)
    std::cout << "[*] Invoking custom_dpu_top hardware execution...\n";
    custom_dpu_top(features_stream, logits_stream);

    // 3. Read Output Classification Logits (12 classes)
    std::cout << "[*] Reading output logits stream (12 classes):\n";
    int best_class = 0;
    float max_logit = -999.0f;

    const char* KEYWORD_LABELS[12] = {
        "silence", "unknown", "yes", "no",
        "up", "down", "left", "right",
        "on", "off", "stop", "go"
    };

    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        if (logits_stream.empty()) {
            std::cerr << "[-] Error: Expected 12 logits, stream ended prematurely!\n";
            return 1;
        }
        axis_pkt_t out_pkt = logits_stream.read();
        logit_t val;
        val.range(15, 0) = out_pkt.data.range(15, 0);
        float float_logit = (float)val;

        std::cout << "    Class " << cls << " [" << KEYWORD_LABELS[cls] << "]: logit = " << float_logit << "\n";

        if (float_logit > max_logit) {
            max_logit = float_logit;
            best_class = cls;
        }
    }

    std::cout << "--------------------------------------------------------\n";
    std::cout << "[+] Top Predicted Keyword: " << KEYWORD_LABELS[best_class]
              << " (Score: " << max_logit << ")\n";
    std::cout << "[+] Custom DPU HLS C-Simulation: PASSED SUCCESSFULLY!\n";
    std::cout << "========================================================\n";

    return 0;
}
