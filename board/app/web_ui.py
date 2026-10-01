"""
board/app/web_ui.py
-------------------
Step 12 — Web UI for Keyword Spotting (KWS) on AMD Kria KV260.

Features:
  - Mode Selection UI:
      [ Passive Data Mode ]    [ Real-Time Voice Mode ]
  - Hardware Engine Selection:
      [ CPU (Cortex-A53) ]     [ DPU (DPUCZDX8G) ]
  - Unified processing: Both modes route into identical preprocessing,
    feature tensor [1, 1, 40, 98], and hardware inference pipeline.
  - Zero external dependencies beyond standard library + NumPy/SciPy/ONNX.
  - Runs on port 8080 (accessible locally or over board network).
"""

from __future__ import annotations

import http.server
import json
import socketserver
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.audio_stream import PassiveAudioLoader, RealTimeAudioCapture
from pipeline.postprocessing import decode
from pipeline.preprocessing import extract_log_mel
from pipeline.utils import SAMPLE_RATE, pad_or_trim

PORT = 8080

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMD Kria KV260 - Audio AI/ML KWS Dashboard</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-green: #22c55e;
            --accent-orange: #f97316;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            width: 100%;
            max-width: 900px;
        }
        header {
            text-align: center;
            margin-bottom: 24px;
        }
        h1 {
            margin: 0 0 8px 0;
            font-size: 26px;
            color: var(--accent);
        }
        p.subtitle {
            color: var(--text-muted);
            margin: 0;
            font-size: 14px;
        }
        .mode-selector {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        .mode-btn {
            background: var(--card-bg);
            border: 2px solid var(--border);
            color: var(--text-muted);
            padding: 16px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .mode-btn.active {
            border-color: var(--accent);
            color: var(--accent);
            background: rgba(56, 189, 248, 0.1);
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .engine-selector {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            align-items: center;
        }
        .engine-label {
            font-size: 14px;
            color: var(--text-muted);
            font-weight: 600;
        }
        select, button.action-btn {
            background: #334155;
            color: var(--text);
            border: 1px solid var(--border);
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
        }
        button.action-btn {
            background: var(--accent);
            color: #0f172a;
            font-weight: bold;
            border: none;
            transition: 0.2s;
        }
        button.action-btn:hover {
            filter: brightness(1.1);
        }
        .result-panel {
            display: none;
            margin-top: 20px;
        }
        .keyword-badge {
            display: inline-block;
            font-size: 32px;
            font-weight: 900;
            color: var(--accent-green);
            background: rgba(34, 197, 94, 0.15);
            padding: 8px 24px;
            border-radius: 10px;
            margin-bottom: 12px;
            letter-spacing: 2px;
        }
        .latency-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 14px;
        }
        .latency-bar {
            height: 10px;
            border-radius: 5px;
            background: #334155;
            margin-top: 12px;
            overflow: hidden;
            display: flex;
        }
        .bar-preproc { background: #38bdf8; }
        .bar-infer { background: #f97316; }
        .bar-post { background: #22c55e; }
        .loader {
            display: none;
            color: var(--accent);
            font-weight: bold;
            text-align: center;
            padding: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AMD Kria KV260 - Audio KWS Accelerator</h1>
            <p class="subtitle">Keyword Spotting on DS-CNN with CPU, DPUCZDX8G, and Custom HLS Acceleration</p>
        </header>

        <!-- Mode Selection -->
        <div class="mode-selector">
            <button class="mode-btn active" id="btn-passive" onclick="setMode('passive')">
                [ 1. Passive Data Mode ]
            </button>
            <button class="mode-btn" id="btn-realtime" onclick="setMode('realtime')">
                [ 2. Real-Time Voice Mode ]
            </button>
        </div>

        <div class="card">
            <!-- Hardware Engine Selector -->
            <div class="engine-selector">
                <span class="engine-label">Hardware Acceleration Engine:</span>
                <select id="engine-select">
                    <option value="cpu">Config A: CPU-Only (ARM Cortex-A53 INT8)</option>
                    <option value="dpu">Config B: CPU + DPU (DPUCZDX8G @ 300MHz)</option>
                </select>
            </div>

            <!-- Passive Mode Control -->
            <div id="panel-passive">
                <label class="engine-label" style="display:block; margin-bottom:8px;">Select Evaluation Audio Clip:</label>
                <div style="display:flex; gap:12px;">
                    <select id="clip-select" style="flex:1;">
                        <option value="data/test_inputs/test_00_yes.wav">test_00_yes.wav (Ground Truth: YES)</option>
                        <option value="data/test_inputs/test_01_yes.wav">test_01_yes.wav (Ground Truth: YES)</option>
                        <option value="data/test_inputs/test_02_no.wav">test_02_no.wav (Ground Truth: NO)</option>
                        <option value="data/test_inputs/test_04_stop.wav">test_04_stop.wav (Ground Truth: STOP)</option>
                        <option value="data/test_inputs/test_06_go.wav">test_06_go.wav (Ground Truth: GO)</option>
                        <option value="data/test_inputs/test_09_silence.wav">test_09_silence.wav (Ground Truth: SILENCE)</option>
                    </select>
                    <button class="action-btn" onclick="runInference()">Analyze Audio</button>
                </div>
            </div>

            <!-- Real-Time Voice Mode Control -->
            <div id="panel-realtime" style="display:none;">
                <p style="color:var(--text-muted); font-size:14px; margin-top:0;">
                    Click the button below to capture 1.0 second of live voice from the microphone.
                    The voice data will be processed through the exact same feature extraction and hardware inference pipeline.
                </p>
                <button class="action-btn" id="record-btn" onclick="captureAndInfer()" style="width:100%; padding:14px;">
                    🎤 Start Speaking (Record 1.0s Voice)
                </button>
            </div>

            <div class="loader" id="loader">Processing audio through pipeline...</div>
        </div>

        <!-- Result Panel -->
        <div class="card result-panel" id="result-panel">
            <div style="text-align:center;">
                <span class="engine-label">Detected Keyword</span><br>
                <div class="keyword-badge" id="res-keyword">YES</div>
                <div style="color:var(--text-muted); font-size:14px;">
                    Confidence Score: <strong id="res-conf" style="color:var(--text);">98.4%</strong> | Class Index: <span id="res-idx">0</span>
                </div>
            </div>

            <h4 style="margin:20px 0 10px 0; color:var(--accent);">Per-Stage Latency Breakdown</h4>
            <div class="latency-row">
                <span>[1] Audio Ingestion (<span id="res-acq-label">WAV IO</span>)</span>
                <strong id="res-load-ms">0.82 ms</strong>
            </div>
            <div class="latency-row">
                <span>[2] Mel Preprocessing (GEMM + Log)</span>
                <strong id="res-preproc-ms">1.25 ms</strong>
            </div>
            <div class="latency-row">
                <span>[3] Neural Inference (<span id="res-eng-label">DPU</span>)</span>
                <strong id="res-infer-ms">1.47 ms</strong>
            </div>
            <div class="latency-row">
                <span>[4] Softmax Postprocessing</span>
                <strong id="res-post-ms">0.05 ms</strong>
            </div>
            <div class="latency-row" style="border-top:1px solid var(--border); font-weight:bold; color:var(--accent);">
                <span>TOTAL CORE PIPELINE LATENCY</span>
                <span id="res-total-ms">2.77 ms (361.0 FPS)</span>
            </div>

            <div class="latency-bar">
                <div class="bar-preproc" id="bar-preproc" style="width: 30%;"></div>
                <div class="bar-infer" id="bar-infer" style="width: 68%;"></div>
                <div class="bar-post" id="bar-post" style="width: 2%;"></div>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'passive';

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('btn-passive').classList.toggle('active', mode === 'passive');
            document.getElementById('btn-realtime').classList.toggle('active', mode === 'realtime');
            document.getElementById('panel-passive').style.display = mode === 'passive' ? 'block' : 'none';
            document.getElementById('panel-realtime').style.display = mode === 'realtime' ? 'block' : 'none';
        }

        async function executeBackend(payload) {
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result-panel').style.display = 'none';

            try {
                const resp = await fetch('/api/infer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
                renderResult(data);
            } catch (err) {
                alert('Inference error: ' + err);
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }

        function runInference() {
            const engine = document.getElementById('engine-select').value;
            const wav = document.getElementById('clip-select').value;
            executeBackend({ mode: 'passive', engine: engine, wav: wav });
        }

        function captureAndInfer() {
            const engine = document.getElementById('engine-select').value;
            executeBackend({ mode: 'realtime', engine: engine });
        }

        function renderResult(data) {
            document.getElementById('res-keyword').innerText = data.keyword.toUpperCase();
            document.getElementById('res-conf').innerText = (data.confidence * 100).toFixed(2) + '%';
            document.getElementById('res-idx').innerText = data.class_idx;

            document.getElementById('res-acq-label').innerText = data.mode === 'passive' ? 'WAV IO' : 'Live Mic';
            document.getElementById('res-load-ms').innerText = data.load_ms.toFixed(2) + ' ms';
            document.getElementById('res-preproc-ms').innerText = data.preproc_ms.toFixed(2) + ' ms';
            document.getElementById('res-infer-ms').innerText = data.infer_ms.toFixed(2) + ' ms';
            document.getElementById('res-eng-label').innerText = data.engine.toUpperCase();
            document.getElementById('res-post-ms').innerText = data.post_ms.toFixed(2) + ' ms';

            const coreTotal = data.preproc_ms + data.infer_ms + data.post_ms;
            const fps = (1000.0 / coreTotal).toFixed(1);
            document.getElementById('res-total-ms').innerText = coreTotal.toFixed(2) + ' ms (' + fps + ' FPS)';

            // Set progress bar proportions
            document.getElementById('bar-preproc').style.width = ((data.preproc_ms / coreTotal) * 100) + '%';
            document.getElementById('bar-infer').style.width = ((data.infer_ms / coreTotal) * 100) + '%';
            document.getElementById('bar-post').style.width = ((data.post_ms / coreTotal) * 100) + '%';

            document.getElementById('result-panel').style.display = 'block';
        }
    </script>
</body>
</html>
"""


class KWSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handles HTTP requests for KWS Web UI."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/infer":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            mode = req.get("mode", "passive")
            engine = req.get("engine", "cpu")
            wav_file = req.get("wav", "data/test_inputs/test_00_yes.wav")

            # 1. Acquire Audio (Passive or Real-Time)
            t0 = time.perf_counter_ns()
            if mode == "passive":
                audio = PassiveAudioLoader.load(ROOT / wav_file)
            else:
                cap = RealTimeAudioCapture()
                audio = cap.record_clip(duration_s=1.0)
            t1 = time.perf_counter_ns()

            # 2. Preprocess (Unified)
            t2 = time.perf_counter_ns()
            features = extract_log_mel(audio, apply_pre_emphasis=True)
            t3 = time.perf_counter_ns()

            # 3. Inference (CPU or DPU)
            t4 = time.perf_counter_ns()
            if engine == "dpu":
                from board.app.dpu_runner import VARTDPURunner
                xmodel_p = ROOT / "models" / "compiled" / "dscnn_medium.xmodel"
                runner = VARTDPURunner(xmodel_p)
                logits, _ = runner.infer(features)
            else:
                from benchmarks.cpu_baseline import CPUModelRunner
                onnx_p = ROOT / "models" / "onnx" / "dscnn_medium.onnx"
                runner = CPUModelRunner(onnx_p)
                logits = runner(features)
            t5 = time.perf_counter_ns()

            # 4. Postprocessing
            t6 = time.perf_counter_ns()
            label, conf, idx = decode(logits)
            t7 = time.perf_counter_ns()

            resp = {
                "keyword": label,
                "confidence": float(conf),
                "class_idx": int(idx),
                "mode": mode,
                "engine": engine,
                "load_ms": (t1 - t0) / 1e6,
                "preproc_ms": (t3 - t2) / 1e6,
                "infer_ms": (t5 - t4) / 1e6,
                "post_ms": (t7 - t6) / 1e6,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def run_server():
    server = socketserver.TCPServer(("", PORT), KWSRequestHandler)
    print("\n" + "=" * 70)
    print(f" AMD Kria KV260 Audio KWS Web UI started on http://localhost:{PORT}")
    print(f" Supporting: [ Passive Data Mode ] & [ Real-Time Voice Mode ]")
    print(" Press Ctrl+C to stop.")
    print("=" * 70 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
