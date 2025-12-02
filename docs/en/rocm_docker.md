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
    AMD’s ROCm support matrix).
- Docker installed and configured.
- Access to GPU devices from Docker (typically `/dev/kfd` and `/dev/dri`).

No additional ROCm setup is required on the host beyond having the AMDGPU
drivers and devices available; the ROCm userspace stack is provided by the
container image.

### 2. ROCm Dockerfile

The ROCm-specific Dockerfile lives at:

- `docker/Dockerfile.rocm`

Key properties:

- Based on an official AMD ROCm PyTorch image, for example:
  - `rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_2.5.1`
  - See the available tags at
    `https://hub.docker.com/r/rocm/pytorch/tags`.
- Installs:
  - System dependencies used by MMDetection3D demos
    (`ffmpeg`, `libsm6`, `libxext6`, `git`, `ninja-build`,
     `libglib2.0-0`, `libxrender-dev`).
  - `openmim`, then `mmengine`, `mmcv>=2.0.0rc4`, `mmdet>=3.0.0`
    using `mim install`.
  - The local `mmdetection3d` repository in editable mode plus
    all dependencies from `requirements.txt`.

The container does **not** build any CUDA/ROCm extensions in this repo,
because `setup.py` currently defines `ext_modules=[]`. Low-level kernels
remain in MMCV/MMDetection, which are installed from prebuilt wheels.

### 3. Building the image

From the repository root:

```bash
docker build -f docker/Dockerfile.rocm -t mmdet3d-rocm .
```

If you want to pin a specific ROCm PyTorch tag, override the build arg:

```bash
docker build \
  --build-arg ROCM_TAG=rocm7.1.1_ubuntu22.04_py3.10_pytorch_2.5.1 \
  -f docker/Dockerfile.rocm \
  -t mmdet3d-rocm .
```

Refer to the tag list on `https://hub.docker.com/r/rocm/pytorch/tags`
for valid values.

### 4. Runtime: exposing AMD GPUs to the container

On a host with AMD GPUs and ROCm-capable drivers, you typically need to
expose:

- `/dev/kfd`
- `/dev/dri`

and ensure the container user is in the `video` group (or otherwise
has permission to use those devices).

Example:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  mmdet3d-rocm \
  python -c "import torch; print(torch.cuda.is_available(), getattr(torch.version, 'hip', None))"
```

On a Fedora laptop with an AMD iGPU, this is often sufficient as long
as:

- The kernel and AMDGPU driver expose `/dev/kfd` and `/dev/dri`.
- Your user is in the `video` group (or the corresponding GPU group on
  your distribution).

If you encounter permission issues, you may need additional Docker
options (e.g. `--security-opt seccomp=unconfined`) as recommended by
AMD’s ROCm container documentation.

### 5. GPU accessibility test script

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

### 6. Inference test script

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

### 7. Known limitations and caveats

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


