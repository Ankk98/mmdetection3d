#!/usr/bin/env python3
"""
Quick Vulkan/WebGPU readiness check for fastplotlib/wgpu.

Note: On newer RDNA GPUs, older Jammy/Mesa stacks may fail RADV init
(VK_ERROR_INITIALIZATION_FAILED) inside containers. If this reports
llvmpipe/OpenGL instead of Vulkan, rebuild on a newer base (e.g. Ubuntu
24.04 with newer Mesa) or run fastplotlib on the host where Vulkan works.

Usage (inside the ROCm container):
    python tools/check_vulkan_fastplotlib.py

What it does:
- Requests a high-performance adapter via wgpu (Rust backend).
- Reports backend type, adapter name, vendor/device IDs, and limits.
- Creates a trivial command submission to ensure the device is usable.

Exit codes:
- 0 on success with Vulkan backend.
- 1 on any failure or if the backend is not Vulkan.
"""

import os
import sys


def main() -> int:
    os.environ.setdefault("WGPU_BACKEND", "vulkan")
    try:
        import wgpu  # noqa: F401
        from wgpu.utils import get_default_device
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] Unable to import wgpu: {exc}")
        return 1

    try:
        # API on wgpu>=0.28: get_default_device() has no power_preference arg
        device = get_default_device()
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] Unable to create device: {exc}")
        return 1

    info = device.adapter.info
    backend = (info.get("backend_type") or info.get("backend") or "unknown")
    name = info.get("name", "unknown")
    vendor = info.get("vendor", "unknown")
    device_id = info.get("device", "unknown")

    print(f"[INFO] WGPU_BACKEND={os.environ.get('WGPU_BACKEND')}")
    print(f"[INFO] Adapter name   : {name}")
    print(f"[INFO] Backend type   : {backend}")
    print(f"[INFO] Vendor / Device: {vendor} / {device_id}")
    print(f"[INFO] Limits         : {device.limits}")

    if backend.lower() != "vulkan":
        print(f"[FAIL] Expected Vulkan backend, got: {backend}")
        return 1

    # Minimal submit to ensure queue works.
    try:
        encoder = device.create_command_encoder()
        device.queue.submit([encoder.finish()])
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] Queue submit failed: {exc}")
        return 1

    print("[OK] Vulkan/WebGPU device is available and queue submit succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
