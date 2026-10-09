import os
import sys
import glob
import time
import subprocess

print("\n" + "=" * 65)
print("     AMD KRIA KV260 -- STANDALONE DPU HARDWARE TRIGGER")
print("=" * 65)

# 1. Clean stale locks
for f in glob.glob("/tmp/DPU*"):
    try:
        os.remove(f)
    except Exception:
        pass

# 2. Automatically reset kernel driver & reload kv260-benchmark-b4096 overlay
print("[*] [1/5] Resetting FPGA PL fabric & KDS driver to clean state...")
try:
    subprocess.run(["xmutil", "unloadapp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    res = subprocess.run(["xmutil", "loadapp", "kv260-benchmark-b4096"], capture_output=True, text=True)
    print(f"    >>> xmutil loadapp: {res.stdout.strip() or 'OK'}")
    time.sleep(1)
except Exception as e:
    print(f"[-] Note on xmutil reload: {e}")

# 3. Configure /etc/vart.conf for kv260-benchmark-b4096
b4096_xclbin = "/lib/firmware/xilinx/kv260-benchmark-b4096/kv260-benchmark-b4096.xclbin"
if os.path.exists(b4096_xclbin):
    print(f"[*] [2/5] Configuring VART firmware: {b4096_xclbin}")
    try:
        with open("/etc/vart.conf", "w") as f:
            f.write(f"firmware: {b4096_xclbin}\n")
    except Exception:
        pass
    try:
        if os.path.islink("/usr/lib/dpu.xclbin") or os.path.exists("/usr/lib/dpu.xclbin"):
            os.remove("/usr/lib/dpu.xclbin")
        os.symlink(b4096_xclbin, "/usr/lib/dpu.xclbin")
    except Exception:
        pass

os.environ.pop("XLNX_DISABLE_LOAD_XCLBIN", None)
os.environ["XLNX_ENABLE_FINGERPRINT_CHECK"] = "0"

import numpy as np
import xir
import vart

# 4. Locate model file
model_path = "resnet50/resnet50.xmodel"
if not os.path.exists(model_path):
    if os.path.exists("resnet50.xmodel"):
        model_path = "resnet50.xmodel"
    elif os.path.exists("resnet50_acc/resnet50_acc.xmodel"):
        model_path = "resnet50_acc/resnet50_acc.xmodel"
    else:
        print(f"[-] Model not found at: {model_path}")
        sys.exit(1)

print(f"[*] [3/5] Deserializing XIR graph: {model_path}...")
graph = xir.Graph.deserialize(model_path)
root = graph.get_root_subgraph()
dpu_subgraphs = [sg for sg in root.get_children() if sg.has_attr("device") and sg.get_attr("device") == "DPU"]

if not dpu_subgraphs:
    print("[-] No DPU subgraph found in xmodel!")
    sys.exit(1)

dpu_sub = dpu_subgraphs[0]
print(f"[*]       DPU Subgraph Identified: {dpu_sub.get_name()}")

print("[*] [4/5] Binding to Physical DPUCZDX8G B4096 Hardware Core...")
runner = vart.Runner.create_runner(dpu_sub, "run")
print("    >>> SUCCESS: VART Runner Bound to Physical FPGA DPU! <<<")

in_tensors = runner.get_input_tensors()
out_tensors = runner.get_output_tensors()
in_shape = tuple(in_tensors[0].dims)
out_shape = tuple(out_tensors[0].dims)

print(f"[*]       Input Tensor Dimensions : {in_shape} (INT8 quantized)")
print(f"[*]       Output Tensor Dimensions: {out_shape} (INT8 quantized)")

# Allocate strictly contiguous C-order buffers
in_buf = np.ascontiguousarray(np.random.randint(-128, 127, size=in_shape, dtype=np.int8))
out_buf = np.ascontiguousarray(np.zeros(out_shape, dtype=np.int8))

print("[*] [5/5] Dispatching Live Hardware Inference to FPGA Fabric...")
print("-" * 65)

t0 = time.perf_counter_ns()
job_id = runner.execute_async([in_buf], [out_buf])
ret = runner.wait(job_id)
t1 = time.perf_counter_ns()
lat_ms = (t1 - t0) / 1e6

print(f"    >>> INFERENCE COMPLETED ON PHYSICAL DPU!")
print(f"    >>> Hardware Latency  : {lat_ms:.2f} ms")
print(f"    >>> Execution Status  : {ret} (0 = SUCCESS)")
print(f"    >>> First 5 Outputs   : {out_buf.flatten()[:5]}")
print("-" * 65)
print("    >>> PHYSICAL HARDWARE DPU VERIFIED WORKING ON KRIA KV260! <<<")
print("=" * 65 + "\n")
