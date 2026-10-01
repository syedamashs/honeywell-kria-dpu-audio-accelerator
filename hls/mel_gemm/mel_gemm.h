#ifndef MEL_GEMM_H_
#define MEL_GEMM_H_

#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// ── Audio Dimensions ─────────────────────────────────────────────────────────
#define N_MELS      40      // Output Mel frequency bands
#define N_BINS      257     // One-sided FFT bins (512-point FFT // 2 + 1)
#define NUM_FRAMES  98      // Frames per 1-second 16kHz clip

// Tiling factors for resource/latency tradeoff on KV260 PL
#define TILE_M      8
#define TILE_K      16

// ── Fixed-Point Data Types ───────────────────────────────────────────────────
// Input power spectrum: non-negative, up to ~100.0, 16-bit word, 8 integer bits
typedef ap_fixed<16, 8, AP_RND, AP_SAT> data_t;

// Mel filterbank weights: normalized [0, 1.0], 16-bit word, 2 integer bits
typedef ap_fixed<16, 2, AP_RND, AP_SAT> weight_t;

// Accumulator: prevent overflow across 257 additions
typedef ap_fixed<32, 12, AP_RND, AP_SAT> acc_t;

// Output: log-mel feature (16-bit)
typedef ap_fixed<16, 6, AP_RND, AP_SAT> out_t;

// ── AXI-Stream Protocol Types ────────────────────────────────────────────────
typedef ap_axis<16, 0, 0, 0> axis_pkt_t;

// ── Top-level HLS Kernel Signature ───────────────────────────────────────────
void mel_gemm_top(
    hls::stream<axis_pkt_t> &power_in,   // Input power spectrum stream: [N_BINS x T]
    hls::stream<axis_pkt_t> &mel_out,    // Output mel stream: [N_MELS x T]
    int num_frames                       // Number of time frames T (default: 98)
);

#endif // MEL_GEMM_H_
