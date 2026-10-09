#include "mel_gemm.h"
// Filterbank weights stored in on-chip BRAM (40 x 257)
// Generated from standard Slaney Auditory Toolbox formula
extern const weight_t MEL_WEIGHTS_ROM[N_MELS][N_BINS];

void mel_gemm_top(
    hls::stream<axis_pkt_t> &power_in,
    hls::stream<axis_pkt_t> &mel_out,
    int num_frames
) {
#pragma HLS INTERFACE axis port=power_in
#pragma HLS INTERFACE axis port=mel_out
#pragma HLS INTERFACE s_axilite port=num_frames bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local ping-pong buffer for one frame of FFT power spectrum (257 bins)
    data_t frame_buf[N_BINS];
#pragma HLS BRAM resource=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=frame_buf cyclic factor=TILE_K dim=1

    // Process frame-by-frame: [T frames]
    FRAME_LOOP: for (int t = 0; t < num_frames; t++) {
#pragma HLS LOOP_TRIPCOUNT min=98 max=98

        // Step 1: Read one frame of 257 FFT bins from AXI-Stream
        READ_FRAME: for (int k = 0; k < N_BINS; k++) {
#pragma HLS PIPELINE II=1
            axis_pkt_t pkt = power_in.read();
            // Reinterpret 16-bit integer representation to ap_fixed
            data_t val;
            val.range(15, 0) = pkt.data.range(15, 0);
            frame_buf[k] = val;
        }

        // Step 2: Mel Filterbank GEMM: Mel[m] = Sum_k (M[m][k] * Frame[k])
        // Compute across all 40 mel bands
        MEL_LOOP: for (int m = 0; m < N_MELS; m++) {
#pragma HLS PIPELINE II=1
            acc_t acc = 0;

            DOT_PRODUCT: for (int k = 0; k < N_BINS; k++) {
                acc += (acc_t)(MEL_WEIGHTS_ROM[m][k] * frame_buf[k]);
            }

            // Step 3: Stream out result with TLAST on the final element
            axis_pkt_t out_pkt;
            out_pkt.data.range(15, 0) = ((out_t)acc).range(15, 0);
            out_pkt.keep = -1;
            out_pkt.strb = -1;
            out_pkt.last = (t == num_frames - 1 && m == N_MELS - 1) ? 1 : 0;
            mel_out.write(out_pkt);
        }
    }
}
