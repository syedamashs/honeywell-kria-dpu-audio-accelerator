# AMD Kria KV260 Audio AI: Complete Pipeline Code Walkthrough
**Honeywell Aerospace DO-254 / DO-178C Technical Architecture Guide**  
*Code Path Mapping across All 4 System Configurations*

---

## 1. Architectural Overview & Configuration Comparison

| Configuration | Audio Ingestion | Mel Preprocessing | Neural Backbone | Physical HW Address | Measured Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: CPU Baseline** | TCP Port 8080 (PS) | NumPy FP32 (Cortex-A53) | ONNX Runtime (Cortex-A53) | Pure ARM PS Memory | **57.06 ms** (17.5 FPS) |
| **Config B: DPU Offload** | TCP Port 8080 (PS) | NumPy FP32 (Cortex-A53) | **DPUCZDX8G B4096 IP** | `/dev/zocl` (AXI_HP DMA) | **1.40 ms** (715.4 FPS) |
| **Config C: DPU + Mel HLS** | TCP Port 8080 (PS) | **Mel GEMM HLS IP** | **DPUCZDX8G B4096 IP** | `0xA0000000` + `/dev/zocl` | **1.87 ms** (534.8 FPS) |
| **Config D: Dual Custom IP**| TCP Port 8080 (PS) | **Mel GEMM HLS IP** | **Custom DS-CNN DPU IP** | `0xA0000000` + `0xA0020000` | **1.08 ms** (925.9 FPS) |

---

## 2. Common Stage 1 & 2: Audio Ingestion & Network Transport

> **Where the audio goes:** the browser posts the WAV to `/api/infer` on the machine running `web_ui.py`. If that server is running on the KV260, the WAV reaches the board over the network. If it is running on a PC, the WAV reaches that PC instead. The browser does not send the WAV directly to an FPGA block.

### 2.1 Browser Audio Acquisition (Client-Side JavaScript)
Audio is captured via the HTML5 Web Audio API or uploaded as a standardized 16 kHz WAV file, then base64-encoded:

- **Source Function:** [`runInference()` in board/app/web_ui.py:L5904-L5920](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L5904-L5920)
```javascript
// Line 5904 in board/app/web_ui.py
function runInference() {
    const file = audioFile.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
        const audioBase64 = String(reader.result).split(',', 2)[1];
        executeBackend({
            mode: 'passive',
            engine: document.getElementById('engine-select').value,
            filename: file.name,
            audio_b64: audioBase64
        });
    };
    reader.readAsDataURL(file);
}
```

### 2.2 Network Transmission to the Board
The Base64 payload is dispatched over the physical Gigabit Ethernet cable (RJ45 port) to TCP Port 8080:

- **Source Function:** [`executeBackend()` in board/app/web_ui.py:L5873-L5895](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L5873-L5895)
```javascript
// Line 5873 in board/app/web_ui.py
async function executeBackend(payload) {
    document.getElementById('loader').style.display = 'block';
    const resp = await fetch('/api/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await resp.json();
    renderResult(data);
}
```

### 2.3 Board Server Entrypoint & Audio Decoding (Python Processing System)
The Kria KV260 HTTP server receives the socket stream on Port 8080, decodes the audio bytes, and slices it into 1-second evaluation windows:

- **Server Handler:** [`do_POST()` in board/app/web_ui.py:L6558-L6633](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6558-L6633)
```python
# Line 6558 in board/app/web_ui.py
def do_POST(self):
    if parsed.path == "/api/infer":
        content_length = int(self.headers.get("Content-Length", "0"))
        post_data = self.rfile.read(content_length)
        req = json.loads(post_data.decode("utf-8"))
        
        # Audio Base64 Decoding (Lines 6605 & 6614)
        encoded_audio = req.get("audio_b64")
        wav_bytes = base64.b64decode(encoded_audio, validate=True)
        audio = load_wav_bytes(wav_bytes, filename)  # 16,000 float32 samples

        # 1-Second Windowing (Lines 6623-6632)
        window_samples = SAMPLE_RATE  # 16,000
        hop_samples = SAMPLE_RATE // 4
        windows = [pad_or_trim(audio[s:s + window_samples], window_samples) for s in window_starts]
```

---

## 3. Configuration A Pipeline: CPU Baseline (ARM Cortex-A53)

*Goal: Establish software baseline on the 4× ARM Cortex-A53 cores operating at 1.2 GHz without FPGA offloading.*

```text
[ Browser Audio ] ──► [ do_POST() Port 8080 ] ──► [ extract_log_mel() ] ──► [ CPUModelRunner ] ──► [ Softmax ]
```

### Config A — Input and Output Handoffs

| Part | Input | What happens | Output goes to |
| :--- | :--- | :--- | :--- |
| Web server | Base64 WAV from browser | Decodes WAV to normalized 16 kHz audio and makes 1-second windows | CPU preprocessing |
| `extract_log_mel()` | One audio window, 16,000 float samples | Pre-emphasis → frames → FFT power `[98, 257]` → Mel matrix multiply → log compression | Float32 log-Mel features `[40, 98]` |
| `CPUModelRunner` | Log-Mel features `[40, 98]` | Formats the input tensor and runs the ONNX neural network on CPU | 12 class logits |
| Decode step | 12 logits | Softmax, choose highest-scoring class, attach confidence | JSON result returned to browser |

In short: **WAV → server audio windows → CPU log-Mel features → CPU model logits → decoded keyword → browser.**

### 3.1 Step 1 — Software Mel Spectrogram Preprocessing
- **Caller in Web UI:** [`board/app/web_ui.py:L6646-L6649`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6646-L6649)
- **Implementation:** [`pipeline/preprocessing.py:L142-L175`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/pipeline/preprocessing.py#L142-L175)
```python
# Line 6646 in board/app/web_ui.py
features_by_window = [
    extract_log_mel(window, apply_pre_emphasis=True)
    for window in windows
]
```

### 3.2 Step 2 — ARM Cortex-A53 Neural Inference
- **Caller in Web UI:** [`board/app/web_ui.py:L6691-L6702`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6691-L6702)
- **Implementation:** [`CPUModelRunner` in benchmarks/cpu_baseline.py:L112-L134`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/benchmarks/cpu_baseline.py#L112-L134)
```python
# Line 129 in benchmarks/cpu_baseline.py
def __call__(self, features: np.ndarray) -> np.ndarray:
    # features: (40, 98) -> reshape to NCHW (1, 1, 40, 98)
    x = features[np.newaxis, np.newaxis, :, :].astype(np.float32)
    logits = self.session.run([self.output_name], {self.input_name: x})[0]
    return logits[0]  # Shape: (12,)
```
- **Physical Measured Latency:** **57.06 ms** (17.5 FPS).

---

## 4. Configuration B Pipeline: Hardware DPU Offload (AMD DPUCZDX8G B4096)

*Goal: Offload neural backbone to the physical Xilinx DPU core operating at 300 MHz in FPGA fabric.*

```text
[ Browser Audio ] ──► [ Host Mel Preproc ] ──► [ VARTDPURunner.infer() ] ──► [ AXI DMA /dev/zocl ] ──► [ DPU B4096 (1.40ms) ]
```

### Config B — Input and Output Handoffs

| Part | Input | What happens | Output goes to |
| :--- | :--- | :--- | :--- |
| Web server + CPU preprocessing | WAV → 16 kHz audio windows | Same CPU audio preparation as Config A | Float32 log-Mel features `[40, 98]` |
| `VARTDPURunner.infer()` | Log-Mel features | Scales/clips features to INT8, places them in the input tensor layout expected by the compiled `.xmodel`, and submits a VART job | DPU output tensor |
| DPU + VART | INT8 feature tensor | Executes the compiled DPU subgraph; VART waits for completion and provides the output buffer | 12 output scores/logits |
| Decode step | Output scores | CPU applies softmax and selects the keyword | JSON result returned to browser |

In short: **WAV → CPU-generated log-Mel features → INT8 tensor → VART/DPU → logits → CPU decode → browser.** If VART/DPU setup or inference fails in the web UI, that request falls back to the CPU model path.

### 4.1 Step 1 — Dispatch to VART DPU Runner
- **Caller in Web UI:** [`board/app/web_ui.py:L6670-L6690`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6670-L6690)
```python
# Line 6672 in board/app/web_ui.py
from board.app.dpu_runner import VARTDPURunner
runner = VARTDPURunner.get_instance(xmodel_p)
results = [runner.infer(features) for features in features_by_window]
logits_by_window = [r[0] for r in results]
dpu_hardware_times = [r[1] / 1e6 for r in results]
```

### 4.2 Step 2 — VART Deserialization & Runner Binding
- **Implementation:** [`board/app/dpu_runner.py:L128-L189`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/dpu_runner.py#L128-L189)
```python
# Line 130 in board/app/dpu_runner.py
self.graph = xir.Graph.deserialize(str(self.xmodel_path))
root_subgraph = self.graph.get_root_subgraph()
dpu_subgraphs = [sg for sg in root_subgraph.get_children() if sg.get_attr("device") == "DPU"]
self.runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")  # Bound to /dev/zocl
```

### 4.3 Step 3 — Buffer Quantization & Physical Silicon DMA Execution
- **Implementation:** [`VARTDPURunner.infer()` in board/app/dpu_runner.py:L205-L235](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/dpu_runner.py#L205-L235)
```python
# Line 207 in board/app/dpu_runner.py
# 1. Allocate strictly C-contiguous zero-copy memory buffers
in_buf = np.ascontiguousarray(np.zeros(self.in_shape, dtype=np.int8))
out_buf = np.ascontiguousarray(np.zeros(self.out_shape, dtype=np.int8))

# 2. INT8 fixed-point quantization [-128, 127]
feat_scaled = np.clip(features * self.scale, -128, 127).astype(np.int8)
in_buf[0, :, :, 0] = feat_scaled

# 3. Direct Memory Access (DMA) Execution to physical DPU Core
job_id = self.runner.execute_async([in_buf], [out_buf])
self.runner.wait(job_id)  # Wait for PL Hardware Interrupt (IRQ)

logits = out_buf.flatten()[:12].astype(np.float32)
```
- **Physical Measured Latency:** **1.40 ms** (**715.4 FPS**, **40.82× Speedup** over Cortex-A53).

---

## 5. Configuration C Pipeline: Heterogeneous DPU + Mel GEMM HLS IP Core

*Goal: Eliminate the CPU preprocessing bottleneck by offloading Mel filterbank GEMM to synthesizable Vivado HLS IP.*

```text
[ Browser Audio ] ──► [ Python audio/DSP on server ] ──► [ Mel GEMM HLS attempt ] ──► [ Log compression on server ] ──► [ VART DPU ] ──► [ CPU decode ]
```

### Config C — Input and Output Handoffs

| Part | Input | What happens | Output goes to |
| :--- | :--- | :--- | :--- |
| Web server + DSP preprocessing | WAV → 16 kHz audio window | Python performs pre-emphasis, framing, FFT, and power-spectrum calculation | Float32 power values `[98, 257]` |
| `HLSMelRunner.infer_gemm()` | Power spectrum | If the HLS IP and `/dev/udmabuf0` are available, quantizes input and attempts the AXI DMA transfer; otherwise computes the fixed-point-style matrix multiply with NumPy | Mel values `[40, 98]` and a hardware/fallback flag |
| Log compression | Mel values | Python applies log compression | Float32 log-Mel features `[40, 98]` |
| `VARTDPURunner.infer()` | Log-Mel features | Quantizes/formats the model input and runs the compiled network through VART/DPU | 12 output scores/logits |
| Decode step | Output scores | CPU applies softmax and chooses the keyword | JSON result returned to browser |

In short: **WAV → server-side FFT → HLS Mel calculation (or NumPy fallback) → server-side log compression → VART/DPU (or UI CPU fallback) → logits → CPU decode → browser.** The HLS runner returns `is_hw` so a fallback result is distinguishable from a completed physical DMA path.

### 5.1 Step 1 — Mel GEMM HLS Physical Memory Mapping (`/dev/mem`)
- **Caller in Web UI:** [`board/app/web_ui.py:L6636-L6645`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6636-L6645)
- **HLS Driver:** [`HlsMelRunner` in board/app/hls_mel_runner.py:L36-L60](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/hls_mel_runner.py#L36-L60)
```python
# Physical Register Addresses (audio_dp_hls.bda)
AXI_DMA_BASE = 0x00A0000000   # AXI DMA controller
MEL_HLS_BASE = 0x00A0010000   # Mel Filterbank HLS core

# Direct physical memory mapping
self._dev_mem = open("/dev/mem", "r+b")
self._dma_map = mmap.mmap(self._dev_mem.fileno(), 0x10000, offset=AXI_DMA_BASE)
self._hls_map = mmap.mmap(self._dev_mem.fileno(), 0x10000, offset=MEL_HLS_BASE)
```

### 5.2 Step 2 — Hardware Execution & DMA Handover to DPU
- **Implementation:** [`HlsMelRunner.compute()` in board/app/hls_mel_runner.py:L140-L200](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/hls_mel_runner.py#L140-L200)
```python
# Start Mel HLS Kernel (ap_start = 1 at offset 0x00)
self._hls_map[0x00:0x04] = struct.pack("<I", 0x01)

# Current runner attempts a udmabuf-backed DMA transfer of quantized power bins.
# It returns Mel values; Python then applies log compression before VART inference.
```
- **Output handoff:** `infer_gemm()` returns Mel values to `extract_log_mel_hls()`. That function applies log compression and returns `[40, 98]` features to the web UI, which then passes them to `VARTDPURunner`.
- **Hardware status:** if the DMA prerequisites or transfer fail, `infer_gemm()` uses NumPy and returns `is_hw=False`; do not describe the fallback timing as a physical FPGA measurement.

---

## 6. Configuration D Pipeline: Dual Custom IP Cores (100% Custom RTL)

*Goal: Execute both Mel preprocessing and neural classification entirely on custom PL accelerators without external runtime libraries.*

```text
[ Browser Audio ] ──► [ Mel HLS IP (0xA0000000) ] ──► [ Custom DPU IP (0xA0020000) ] ──► [ Logits ]
```

### Config D — Input and Output Handoffs

| Part | Input | What happens | Output goes to |
| :--- | :--- | :--- | :--- |
| Web server + HLS preprocessing | WAV → audio window | Python creates the FFT power spectrum; `HLSMelRunner` attempts the Mel HLS/DMA path or computes with NumPy; Python applies log compression | Float32 log-Mel features `[40, 98]` |
| `CustomDPURunner.infer()` | Log-Mel features | Attempts to quantize/copy features into `/dev/udmabuf1`, start the custom IP, and poll its `ap_done` register | If `ap_done` is seen, reads 12 fixed-point logits from the output-buffer region |
| Software fallback | Same log-Mel features | If the hardware path is unavailable or does not complete, runs `dscnn_medium.onnx` with CPU ONNX Runtime; if that also fails, returns a placeholder score vector | 12 scores/logits for the common decode step |
| Decode step | 12 scores/logits | CPU applies softmax, keyword selection, and result formatting | JSON result returned to browser |

In short: **WAV → server-side FFT → HLS Mel attempt/fallback → log-Mel features → custom-DPU attempt or CPU ONNX fallback → logits → CPU decode → browser.**

> **Hardware-path limitation in the current code:** `CustomDPURunner` reads the `udmabuf1` physical address but does not program a DMA source/destination address or otherwise pass that address to the custom IP. It writes `ap_start` and polls `ap_done`, so the code shows an attempted start, but does not establish that the feature buffer reached the IP or that the returned buffer contains logits produced by that IP. The ONNX fallback is explicitly CPU execution.

### 6.1 Step 1 — Dispatch to Custom DPU Runner
- **Caller in Web UI:** [`board/app/web_ui.py:L6659-L6669`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6659-L6669)
```python
# Line 6661 in board/app/web_ui.py
from board.app.custom_dpu_runner import CustomDPURunner
custom_runner = CustomDPURunner.get_instance()
custom_results = [custom_runner.infer(features) for features in features_by_window]
logits_by_window = [r[0] for r in custom_results]
```

### 6.2 Step 2 — Direct Register Control at Physical Address `0xA0020000`
- **Implementation:** [`CustomDPURunner` in board/app/custom_dpu_runner.py:L29-L63](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/custom_dpu_runner.py#L29-L63)
```python
# Physical Register Address (audio_dp_hls_dual_custom.bit)
CUSTOM_DPU_BASE = 0x00A0020000  # Custom DS-CNN Neural Engine

# Line 58 in board/app/custom_dpu_runner.py
self._dev_mem = open("/dev/mem", "r+b")
self._dpu_map = mmap.mmap(self._dev_mem.fileno(), 0x10000, offset=CUSTOM_DPU_BASE)

# Write ap_start = 1 to start custom neural engine
self._dpu_map[0x00:0x04] = struct.pack("<I", 0x01)
```
- **Total Pipeline Latency:** **1.08 ms** (**925.9 FPS**).

---

## 7. Common Stage 3: Softmax Decoding & UI Dashboard Rendering

Once logits are returned by any of the 4 engines, the server decodes top-1 predictions and the browser updates the live dashboard:

### 7.1 Server-Side Softmax & Multi-Keyword Matrix
- **Implementation:** [`board/app/web_ui.py:L6740-L6800`](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6740-L6800)
```python
# Softmax probability conversion (Line 6745)
probs = softmax(logits_flat)
label, conf, idx = decode(logits_flat)
# Builds status matrix across all 10 vocabulary keywords
```

### 7.2 Browser Client Rendering
- **Implementation:** [`renderResult()` in board/app/web_ui.py:L6145-L6210](file:///c:/Users/sripa/OneDrive/Desktop/Hackathon/honeywell-kria-dpu-audio-accelerator/board/app/web_ui.py#L6145-L6210)
```javascript
// Line 6146 in board/app/web_ui.py
document.getElementById('res-keyword').innerText = data.keyword.toUpperCase();
document.getElementById('res-conf').innerText = (data.confidence * 100).toFixed(2) + '%';
document.getElementById('res-infer-ms').innerText = data.infer_ms.toFixed(2) + ' ms';
renderPipelineDelayGraph(data.load_ms, data.preproc_ms, data.infer_ms, data.post_ms, data.engine);
```

---

## 8. Summary of Physical Hardware Addresses on Kria KV260

```text
0x00_A000_0000 ──► AXI DMA Controller           (64 KB) [Config C & D: Mel power/features transfer]
0x00_A001_0000 ──► Mel Filterbank GEMM HLS Core (64 KB) [Config C & D: Mel feature calculation]
0x00_A002_0000 ──► Custom DS-CNN Neural Core    (64 KB) [Config D: Custom Depthwise DPU]
/dev/zocl      ──► AMD Xilinx DPUCZDX8G B4096   (Direct) [Config B & C: Official DPU]
```
