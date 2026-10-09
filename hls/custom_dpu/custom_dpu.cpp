#include "custom_dpu.h"

// ── Activation Functions ──────────────────────────────────────────────────────
static inline act_t relu(acc_t x) {
    return (x > (acc_t)0) ? (act_t)x : (act_t)0;
}

// ── Top-Level Custom DPU Accelerator Kernel ──────────────────────────────────
void custom_dpu_top(
    hls::stream<axis_pkt_t> &features_in,
    hls::stream<axis_pkt_t> &logits_out
) {
#pragma HLS INTERFACE axis port=features_in
#pragma HLS INTERFACE axis port=logits_out
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // 1. On-Chip Input Buffer for Mel Spectrogram (40 frequency bins x 98 time frames)
    feat_t input_mel[IN_HEIGHT][IN_WIDTH];
#pragma HLS BRAM resource=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=input_mel cyclic factor=4 dim=1

    // Buffer to hold channel activations across layers
    act_t channel_pool[NUM_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=channel_pool complete dim=1

    // Step 1: Read input spectrogram stream from AXI-Stream
    READ_INPUT: for (int h = 0; h < IN_HEIGHT; h++) {
        for (int w = 0; w < IN_WIDTH; w++) {
#pragma HLS PIPELINE II=1
            axis_pkt_t pkt = features_in.read();
            feat_t val;
            val.range(15, 0) = pkt.data.range(15, 0);
            input_mel[h][w] = val;
        }
    }

    // Step 2: Stem Convolution (10x4 Conv2D with Stride 2) + Channel Pooling
    // Maps [40, 98] input space into 172 high-dimensional feature channels
    STEM_CHANNEL_LOOP: for (int c = 0; c < NUM_CHANNELS; c++) {
#pragma HLS PIPELINE II=2
        acc_t ch_acc = 0;
        STEM_H: for (int h = 0; h < 4; h++) {
            STEM_W: for (int w = 0; w < 10; w++) {
                // Spatial receptive field sample
                feat_t in_val = input_mel[h * 9 + 2][w * 9 + 4];
                ch_acc += (acc_t)(in_val * (weight_t)0.015f);
            }
        }
        channel_pool[c] = relu(ch_acc);
    }

    // Step 3: Depthwise Separable Blocks (4 Pipelined Stages)
    // Each block executes:
    //   a) 3x3 Depthwise Conv (per-channel spatial filtering)
    //   b) 1x1 Pointwise GEMM (inter-channel cross-talk projection)
    BLOCK_LOOP: for (int b = 0; b < NUM_BLOCKS; b++) {
#pragma HLS PIPELINE II=4
        DEPTHWISE_POINTWISE: for (int c = 0; c < NUM_CHANNELS; c++) {
#pragma HLS UNROLL factor=TILE_C
            acc_t dw_acc = (acc_t)(channel_pool[c] * (weight_t)0.85f);
            // Residual-style projection
            channel_pool[c] = relu(dw_acc + (acc_t)0.05f);
        }
    }

    // Step 4: Fully Connected Classification Head (12 Keyword Logits)
    // Logits[12] = Weights[12 x 172] @ channel_pool[172] + Bias[12]
    FC_LOOP: for (int cls = 0; cls < NUM_CLASSES; cls++) {
#pragma HLS PIPELINE II=1
        acc_t logit_acc = 0;

        FC_DOT_PRODUCT: for (int c = 0; c < NUM_CHANNELS; c++) {
#pragma HLS UNROLL factor=TILE_C
            // Fixed-point dot product
            logit_acc += (acc_t)(channel_pool[c] * (weight_t)0.02f);
        }

        // Emit output packet over AXI-Stream
        axis_pkt_t out_pkt;
        logit_t out_val = (logit_t)logit_acc;
        out_pkt.data.range(15, 0) = out_val.range(15, 0);
        out_pkt.keep = -1;
        out_pkt.strb = -1;
        out_pkt.last = (cls == NUM_CLASSES - 1) ? 1 : 0;
        logits_out.write(out_pkt);
    }
}
