#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Environment verification script for MMDetection3D ROCm setup.

This script checks the environment setup including:
- Python version and environment variables
- PyTorch and ROCm/HIP support
- MMDetection3D dependencies (mmcv, mmengine, mmdet, mmdet3d)
- Open3D and other key packages
- System tools availability

Usage:
    python tools/verify_rocm_env.py
"""
import os
import subprocess
import sys


def main():
    print("=" * 80)
    print("MMDetection3D ROCm Environment Verification")
    print("=" * 80)

    # Python version
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    # Environment variables
    print("\n" + "-" * 80)
    print("Environment Variables:")
    print("-" * 80)
    env_vars = [
        "PYTHONPATH",
        "ROCM_HOME",
        "HIP_PLATFORM",
        "FORCE_CUDA",
        "PATH",
    ]
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        if var == "PATH":
            # Truncate PATH for readability
            paths = value.split(":")
            rocm_paths = [p for p in paths if "rocm" in p.lower()]
            print(f"  {var}: {len(paths)} total paths, {len(rocm_paths)} ROCm paths")
            for p in rocm_paths[:3]:  # Show first 3 ROCm paths
                print(f"    - {p}")
        else:
            print(f"  {var}: {value}")

    # PyTorch
    print("\n" + "-" * 80)
    print("PyTorch Information:")
    print("-" * 80)
    try:
        import torch
        print(f"  Version: {torch.__version__}")
        print(f"  HIP Version: {getattr(torch.version, 'hip', 'N/A')}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  Note: GPU not available. Check at runtime with GPU devices.")
    except ImportError as e:
        print(f"  ERROR: Failed to import torch: {e}")

    # MMCV
    print("\n" + "-" * 80)
    print("MMCV Information:")
    print("-" * 80)
    try:
        import mmcv
        print(f"  Version: {mmcv.__version__}")
        from mmcv.utils.env import collect_env
        env_info = collect_env()
        compiler = env_info.get("MMCV CUDA Compiler", "N/A")
        print(f"  Compiler: {compiler}")
        if "hip" in compiler.lower() or "rocm" in compiler.lower():
            print("  ✓ ROCm/HIP support detected")
        else:
            print("  ⚠ Warning: ROCm/HIP support may not be properly configured")
    except ImportError as e:
        print(f"  ERROR: Failed to import mmcv: {e}")

    # MMEngine
    print("\n" + "-" * 80)
    print("MMEngine Information:")
    print("-" * 80)
    try:
        import mmengine
        print(f"  Version: {mmengine.__version__}")
    except ImportError as e:
        print(f"  ERROR: Failed to import mmengine: {e}")

    # MMDetection
    print("\n" + "-" * 80)
    print("MMDetection Information:")
    print("-" * 80)
    try:
        import mmdet
        print(f"  Version: {mmdet.__version__}")
    except ImportError as e:
        print(f"  ERROR: Failed to import mmdet: {e}")

    # MMDetection3D
    print("\n" + "-" * 80)
    print("MMDetection3D Information:")
    print("-" * 80)
    try:
        import mmdet3d
        print(f"  Version: {mmdet3d.__version__}")
        print(f"  Location: {mmdet3d.__file__}")
    except ImportError as e:
        print(f"  ERROR: Failed to import mmdet3d: {e}")

    # Open3D
    print("\n" + "-" * 80)
    print("Open3D Information:")
    print("-" * 80)
    try:
        import open3d as o3d
        print(f"  Version: {o3d.__version__}")
    except ImportError as e:
        print(f"  ERROR: Failed to import open3d: {e}")

    # Other key packages
    print("\n" + "-" * 80)
    print("Other Key Packages:")
    print("-" * 80)
    packages = ["numpy", "gdown", "tensorboard"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")

    # System tools
    print("\n" + "-" * 80)
    print("System Tools:")
    print("-" * 80)
    tools = ["screen", "git", "cmake"]
    for tool in tools:
        try:
            result = subprocess.run(
                ["which", tool],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  {tool}: {result.stdout.strip()}")
            else:
                print(f"  {tool}: NOT FOUND")
        except Exception:
            print(f"  {tool}: CHECK FAILED")

    # Working directory
    print("\n" + "-" * 80)
    print("Working Directory:")
    print("-" * 80)
    cwd = os.getcwd()
    print(f"  Current: {cwd}")
    expected_paths = ["/mmdetection3d", "/workspace/mmdetection3d"]
    if cwd in expected_paths:
        print(f"  ✓ Working directory is correct")
    else:
        print(f"  Expected: /mmdetection3d or /workspace/mmdetection3d")

    print("\n" + "=" * 80)
    print("Verification Complete")
    print("=" * 80)
    print("\nNote: GPU availability can only be verified at runtime with proper device access.")
    print("Run 'python tools/check_rocm_gpu.py' to verify GPU access.")


if __name__ == "__main__":
    main()
