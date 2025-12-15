## Running MMDetection3D on AMD GPUs via ROCm

This document describes how to build and run a Docker image for MMDetection3D
on AMD GPUs using ROCm 7.1.1 and the official `rocm/pytorch` base images
from AMD (`https://hub.docker.com/r/rocm/pytorch/tags`).

The image is designed primarily for **inference** but can often be reused
for training with appropriate volume mounts and runtime options.

### 1. Prerequisites

- A Linux host with:
  - Recent kernel and AMDGPU drivers.
  - ROCm 7.1.1–compatible hardware (for laptop iGPU this will depend on
    AMD's ROCm support matrix).
- Docker installed and configured (Docker Compose optional but recommended).
- Access to GPU devices from Docker (typically `/dev/kfd` and `/dev/dri`).

No additional ROCm setup is required on the host beyond having the AMDGPU
drivers and devices available; the ROCm userspace stack is provided by the
container image.

### 2. ROCm Dockerfile

The ROCm-specific Dockerfile lives at:

- `docker/Dockerfile.rocm`

Key properties:

- Based on an official AMD ROCm PyTorch image, default:
  - `rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1`
  - See the available tags at
    `https://hub.docker.com/r/rocm/pytorch/tags`.
- Uses a **multi-stage build**:
  - Stage 1 (`mmcv-builder`): Builds MMCV from source with ROCm support and
    caches the wheel file to avoid recompilation on subsequent builds.
  - Stage 2 (main): Installs dependencies and MMDetection3D.
- Installs:
  - System dependencies:
    - MMDetection3D demo dependencies (`ffmpeg`, `libsm6`, `libxext6`,
      `libglib2.0-0`, `libxrender-dev`).
    - Build tools (`git`, `ninja-build`, `build-essential`, `cmake`).
    - Utilities (`xvfb` for headless GUI, `screen`, `vim`).
  - Python packages:
    - `mmengine` (pure Python).
    - `mmcv` (built from source with ROCm support in the builder stage).
    - `mmdet>=3.0.0,<3.3.0`.
    - MMDetection3D dependencies (`numpy<2.0`, `open3d`, `nuscenes-devkit`,
      `jupyterlab`, `notebook`, `tensorboard`, etc.).
    - The local `mmdetection3d` repository in editable mode.
- Includes an entrypoint script (`docker/docker-entrypoint-rocm.sh`) that
  automatically runs environment verification on container startup (unless
  `SKIP_VERIFY=1` is set).

The container does **not** build any CUDA/ROCm extensions in this repo,
because `setup.py` currently defines `ext_modules=[]`. Low-level kernels
remain in MMCV/MMDetection, which are built from source with ROCm support
during the Docker build process.

### 3. Building the image

From the repository root:

```bash
docker build -f docker/Dockerfile.rocm -t mmdet3d-rocm .
```

The build uses a multi-stage process:
1. First stage builds MMCV from source with ROCm support and caches the wheel.
2. Second stage installs all dependencies and MMDetection3D.

**Note**: The first build may take longer as it compiles MMCV. Subsequent builds
will reuse the cached MMCV wheel unless the MMCV version changes.

If you want to pin a specific ROCm PyTorch tag, override the build arg:

```bash
docker build \
  --build-arg ROCM_TAG=rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1 \
  -f docker/Dockerfile.rocm \
  -t mmdet3d-rocm .
```

Refer to the tag list on `https://hub.docker.com/r/rocm/pytorch/tags`
for valid values.

### 4. Running the container

#### Option A: Using Docker Compose (Recommended)

A `docker-compose.rocm.yml` file is provided for convenient setup:

```bash
docker-compose -f docker-compose.rocm.yml up -d
```

This configuration:
- Automatically exposes GPU devices (`/dev/kfd`, `/dev/dri`).
- Adds the container to the `video` group.
- Mounts the project directory, data, checkpoints, and work directories.
- Sets up environment variables (ROCm paths, HIP platform, etc.).
- Exposes ports for Jupyter (8889) and Open3D WebRTC (8888).
- Runs environment verification on startup (unless `SKIP_VERIFY=1`).

To enter the container:

```bash
docker-compose -f docker-compose.rocm.yml exec mmdet3d-rocm bash
```

To stop the container:

```bash
docker-compose -f docker-compose.rocm.yml down
```

#### Option B: Using Docker run

On a host with AMD GPUs and ROCm-capable drivers, you typically need to
expose:

- `/dev/kfd`
- `/dev/dri`

and ensure the container user is in the `video` group (or otherwise
has permission to use those devices).

Example:

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -v $(pwd):/workspace/mmdetection3d \
  -v $(pwd)/data:/workspace/mmdetection3d/data \
  -v $(pwd)/checkpoints:/workspace/mmdetection3d/checkpoints \
  -w /workspace/mmdetection3d \
  mmdet3d-rocm \
  bash
```

On a Fedora laptop with an AMD iGPU, this is often sufficient as long
as:

- The kernel and AMDGPU driver expose `/dev/kfd` and `/dev/dri`.
- Your user is in the `video` group (or the corresponding GPU group on
  your distribution).

If you encounter permission issues, you may need additional Docker
options (e.g. `--security-opt seccomp=unconfined`) as recommended by
AMD's ROCm container documentation.

**Note**: The entrypoint script automatically runs environment verification
on startup. To skip verification, set `SKIP_VERIFY=1` in the environment.

### 5. Environment verification

The container includes a comprehensive environment verification script that
checks:

- Python version and environment variables.
- PyTorch and ROCm/HIP support.
- MMDetection3D dependencies (mmcv, mmengine, mmdet, mmdet3d).
- Open3D and other key packages.
- System tools availability.

The verification script runs automatically on container startup via the
entrypoint script. You can also run it manually:

```bash
python tools/verify_rocm_env.py
```

To skip automatic verification on startup, set `SKIP_VERIFY=1`:

```bash
docker run -e SKIP_VERIFY=1 ... mmdet3d-rocm
```

or in `docker-compose.rocm.yml`:

```yaml
environment:
  - SKIP_VERIFY=1
```

### 6. GPU accessibility test script

To quickly verify that the container can see your AMD GPU through ROCm,
a helper script is provided:

- `tools/check_rocm_gpu.py`

Run inside the container:

```bash
python tools/check_rocm_gpu.py
```

This script prints:

- Whether `torch.cuda.is_available()` is `True`.
- Number of visible devices.
- Device names.
- Basic tensor operations on `cuda:0` (HIP-backed) to ensure kernels run.

If this script reports no CUDA/HIP devices, double-check:

- That `/dev/kfd` and `/dev/dri` are passed to the container.
- That your user has permissions to use those devices.

### 7. Inference test script

To verify a full inference path with MMDetection3D, a simple CLI wrapper
around the high-level APIs is provided:

- `tools/test_rocm_inference.py`

Usage (inside the container):

```bash
python tools/test_rocm_inference.py \
  --config configs/second/hv_second_secfpn_4x8_80e_kitti-3d-3class.py \
  --checkpoint /path/to/your_checkpoint.pth \
  --pcd /path/to/example_pointcloud.bin
```

This script:

- Loads a model using `mmdet3d.apis.init_model` on device `cuda:0`
  (which is ROCm-backed under the hood).
- Runs a single inference using `mmdet3d.apis.inference_detector`.
- Prints a brief summary of the output (number of detected boxes per
  class) to confirm that data flows end‑to‑end.

You are responsible for providing:

- A valid MMDetection3D config.
- A matching checkpoint.
- At least one sample point cloud (or dataset-specific input) that
  matches the config’s data pipeline.

On a Fedora laptop with an AMD iGPU, you can use a small test sample
from any supported dataset (e.g. KITTI, nuScenes) copied into a local
directory and mounted into the container via `-v`.

### 8. Known limitations and caveats

- ROCm support depends on your specific AMD GPU/iGPU and driver stack.
- Some MMCV/MMDetection custom CUDA kernels may not have ROCm-optimized
  equivalents:
  - They may fall back to CPU implementations (slower).
  - In rare cases they may not be available at all, causing runtime
    errors.
- Visualization/backends such as `open3d` are expected to run in CPU
  mode unless you build ROCm-aware binaries yourself.

For production use, test a few representative models and workloads on
your specific hardware and consider pinning:

- The ROCm PyTorch base tag (`ROCM_TAG`).
- Versions of `mmengine`, `mmcv`, `mmdet` and this repository.


