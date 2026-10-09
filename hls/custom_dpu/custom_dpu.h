#ifndef CUSTOM_DPU_H_
#define CUSTOM_DPU_H_

#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// ── Model Dimensions (DS-CNN Medium) ──────────────────────────────────────────
#define IN_CHANNELS     1
#define IN_HEIGHT       40      // 40 Mel frequency bands
#define IN_WIDTH        98      // 98 Time frames

#define NUM_CHANNELS    172     // Backbone hidden channels
#define NUM_BLOCKS      4       // 4 Depthwise Separable Conv blocks
#define NUM_CLASSES     12      // 12 Keyword Spotting classes

// Hardware Tiling for Kria KV260 DSP48E2 allocation
#define TILE_C          16      // Channel parallel unroll factor
#define TILE_K          8       // Kernel parallel factor

// ── Fixed-Point Data Types ───────────────────────────────────────────────────
// Input log-mel features: normalized [-10.0, 10.0], 16-bit word, 6 integer bits
typedef ap_fixed<16, 6, AP_RND, AP_SAT> feat_t;

// Weight tensors: normalized [-2.0, 2.0], 16-bit word, 2 integer bits
typedef ap_fixed<16, 2, AP_RND, AP_SAT> weight_t;

// Intermediate feature maps after ReLU
typedef ap_fixed<16, 6, AP_RND, AP_SAT> act_t;

// High-precision accumulator to prevent overflow during sum of products
typedef ap_fixed<32, 12, AP_RND, AP_SAT> acc_t;

// Output logits: 16-bit signed
typedef ap_fixed<16, 6, AP_RND, AP_SAT> logit_t;

// ── AXI4-Stream Interface Protocol ───────────────────────────────────────────
typedef ap_axis<16, 0, 0, 0> axis_pkt_t;

// ── Top-Level Custom DPU Kernel Signature ─────────────────────────────────────
void custom_dpu_top(
    hls::stream<axis_pkt_t> &features_in,   // Ingests [40 x 98] Mel spectrogram stream
    hls::stream<axis_pkt_t> &logits_out     // Emits 12 classification logits stream
);

#endif // CUSTOM_DPU_H_
