# Complete Docker Deployment Guide for AMD Kria KV260
### Keyword Spotting (KWS) Audio Accelerator
**Honeywell Aerospace Hackathon**

---

## 1. Why Docker on the AMD Kria KV260?

On embedded boards like the **AMD Kria KV260 Starter Kit**, running `pip install` directly causes major issues:
1. **No Internet in the Lab**: Hardware testbenches and evaluation rooms often have no internet or restricted network access.
2. **Slow / Failed Compilations**: The KV260 quad-core ARM Cortex-A53 processor has limited RAM (4 GB). Compiling packages like `numpy`, `scipy`, or `onnxruntime` natively can take 45+ minutes or crash with Out-Of-Memory (OOM) errors.
3. **Missing System Libraries**: Packages like `soundfile` require native C libraries (`libsndfile1`). If they are missing from the board's Ubuntu OS, `pip` crashes.
4. **Dirty Root Filesystem**: Multiple team members running `sudo pip` can corrupt the system Python environment.

### The Docker Solution:
Docker packages **everything** into a single frozen snapshot (called a **Docker Image**):
- ✅ The lightweight Linux OS (`python:3.10-slim`)
- ✅ All system audio libraries (`libsndfile1`, `libportaudio2`)
- ✅ All pre-installed Python packages (`numpy`, `scipy`, `soundfile`, `onnx`, `onnxruntime`)
- ✅ All project models (`models/onnx/dscnn_medium.onnx`, INT8 weights)
- ✅ Audio test data clips (`data/test_inputs/`)
- ✅ The interactive Web Dashboard & benchmarks (`board/app/web_ui.py`)

On the board, you **never run `pip`**. You just run **one single command** and the entire application starts in seconds!

---

## 2. What Is Inside This Project?

We have prepared all the necessary files in your repository:

| File | Purpose |
| :--- | :--- |
| [`Dockerfile`](Dockerfile) | The blueprint recipe that builds the complete container. |
| [`requirements-docker.txt`](requirements-docker.txt) | Streamlined runtime Python packages without heavy dev bloat. |
| [`.dockerignore`](.dockerignore) | Excludes temporary files, git history, and caches to keep the image fast & small. |
| [`docker_build.bat`](docker_build.bat) | Windows 1-click builder script. |
| [`docker_build.sh`](docker_build.sh) | Linux/Bash builder script. |

---

## 3. How to Build the Docker Image

### Step 1: Install Docker Desktop (on your Windows PC)
If Docker is not already installed on your PC:
1. Download **Docker Desktop for Windows** from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
2. Run the installer and ensure **WSL 2** is enabled.
3. Start Docker Desktop and wait until the whale icon says "Engine running".

### Step 2: Build the Image
Open PowerShell or Command Prompt in this project folder and run:
```powershell
docker build -t kria-kv260-kws:latest .
```
*(Or simply double-click [`docker_build.bat`](docker_build.bat)).*

### Step 3: Test It Locally on Your PC
Verify the container works before sending it anywhere:
```powershell
docker run -p 8080:8080 kria-kv260-kws:latest
```
Now open your browser to **`http://localhost:8080`**.
You will see the full Honeywell KWS Web Dashboard running from inside the container! Press `Ctrl+C` in your terminal when done.

---

## 4. Where Do We Put the Image?

You have **two options** to deploy this image to the board:

---

### Option A: Push to Docker Hub (Recommended if the Board has Internet / Wi-Fi)

**Docker Hub** is the official cloud registry for Docker images (like GitHub, but for Docker).

1. **Create a Free Account**:
   - Go to [https://hub.docker.com/](https://hub.docker.com/) and register (e.g., username `myteam`).

2. **Log In on Your PC Terminal**:
   ```bash
   docker login
   ```
   *(Enter your Docker Hub username and password).*

3. **Tag Your Image with Your Docker Hub Username**:
   ```bash
   docker tag kria-kv260-kws:latest <YOUR_USERNAME>/kria-kv260-kws:latest
   ```
   *(Example: `docker tag kria-kv260-kws:latest amash/kria-kv260-kws:latest`)*

4. **Push to Docker Hub**:
   ```bash
   docker push <YOUR_USERNAME>/kria-kv260-kws:latest
   ```

5. **Run on the KV260 Board**:
   SSH into the KV260 board terminal and run:
   ```bash
   docker run -d -p 8080:8080 --name kws-app <YOUR_USERNAME>/kria-kv260-kws:latest
   ```
   The board downloads the image and launches it automatically!

---

### Option B: Offline `.tar` File (Best for Lab Demos with NO Internet!)

If the KV260 board in your hackathon booth does **not** have internet access, you don't even need Docker Hub! You can export the entire container as an offline file.

1. **Export the Image to a File on Your PC**:
   ```powershell
   docker save -o kv260_kws_image.tar kria-kv260-kws:latest
   ```
   This generates a single file `kv260_kws_image.tar`.

2. **Copy the File to the KV260 Board**:
   - Via **USB Pen Drive**: Plug pen drive into PC, copy `kv260_kws_image.tar`, plug into KV260.
   - OR via **SCP** (Ethernet cable):
     ```powershell
     scp kv260_kws_image.tar ubuntu@<KV260_IP_ADDRESS>:~/
     ```

3. **Load the Image on the KV260 Board**:
   On the KV260 board terminal:
   ```bash
   docker load -i kv260_kws_image.tar
   ```

4. **Run the Container on the Board**:
   ```bash
   docker run -d -p 8080:8080 --name kws-app kria-kv260-kws:latest
   ```
   **Zero internet needed! Zero `pip` commands needed!**

---

## 5. Important: Building Specifically for ARM64 (Kria KV260 Architecture)

The KV260 board has an **ARM64 (aarch64)** processor, while your Windows PC is likely **x86_64**.
Docker has a built-in feature called **Buildx** that builds ARM64 images directly from your Windows PC:

```powershell
# 1. Enable multi-platform builder
docker buildx create --use

# 2. Build directly for ARM64 (for the board) and push to Docker Hub
docker buildx build --platform linux/arm64 -t <YOUR_USERNAME>/kria-kv260-kws:latest --push .
```
Or to save directly as an ARM64 tar file:
```powershell
docker buildx build --platform linux/arm64 -t kria-kv260-kws:latest --output type=docker -o kv260_arm64.tar .
```

---

## 6. How to Access the Running App on the Board

Once running on the KV260 with:
```bash
docker run -d -p 8080:8080 --name kws-app kria-kv260-kws:latest
```

1. Find the board's IP address:
   ```bash
   hostname -I
   ```
   *(For example: `192.168.1.150`)*

2. Open any browser on your laptop (connected to the same Wi-Fi / Ethernet):
   ```
   http://192.168.1.150:8080
   ```
3. You will see the live interactive dashboard:
   - Select audio files or speak live into the microphone.
   - Run Config A (CPU Baseline) or staged Config B/C (DPU + HLS).
   - View latency telemetry and prediction charts!

---

## 7. Useful Docker Commands Quick Reference

| Action | Command |
| :--- | :--- |
| **Check running containers** | `docker ps` |
| **View live container logs** | `docker logs -f kws-app` |
| **Stop the container** | `docker stop kws-app` |
| **Restart the container** | `docker restart kws-app` |
| **Enter container interactive shell** | `docker exec -it kws-app bash` |
| **Remove container** | `docker rm -f kws-app` |
