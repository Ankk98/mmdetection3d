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
from contextlib import redirect_stderr
from io import StringIO


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
        # Suppress stderr from collect_env() to avoid hipcc not found errors
        # (hipcc is only needed for building, not runtime)
        stderr_buffer = StringIO()
        with redirect_stderr(stderr_buffer):
            env_info = collect_env()
        compiler = env_info.get("MMCV CUDA Compiler", "N/A")
        print(f"  Compiler Info: {compiler}")
        
        # Check for ROCm/HIP support
        # The compiler string might be a version number (e.g., "70152802" = ROCm 7.1.52802)
        # or contain "hip" or "rocm" in the path
        compiler_lower = str(compiler).lower()
        has_hip_version = False
        try:
            # Check if it's a numeric version that matches ROCm version pattern
            if compiler.isdigit() and len(compiler) >= 6:
                # ROCm versions are typically 8+ digits (e.g., 70152802)
                has_hip_version = True
        except (AttributeError, ValueError):
            pass
        
        if "hip" in compiler_lower or "rocm" in compiler_lower or has_hip_version:
            print("  ✓ ROCm/HIP support detected")
        else:
            # Check PyTorch HIP support as fallback
            try:
                import torch
                if hasattr(torch.version, 'hip') and torch.version.hip:
                    print("  ✓ ROCm/HIP support detected (via PyTorch)")
                else:
                    print("  ⚠ Warning: ROCm/HIP support may not be properly configured")
            except Exception:
                print("  ⚠ Warning: ROCm/HIP support may not be properly configured")
    except ImportError as e:
        print(f"  ERROR: Failed to import mmcv: {e}")
    
    # Check for hipcc compiler
    print("\n" + "-" * 80)
    print("ROCm Tools:")
    print("-" * 80)
    hipcc_paths = [
        "/opt/rocm/hip/bin/hipcc",
        "/opt/rocm/bin/hipcc",
        "/usr/bin/hipcc",
    ]
    hipcc_found = False
    for path in hipcc_paths:
        if os.path.exists(path):
            print(f"  hipcc: {path} ✓")
            hipcc_found = True
            break
    if not hipcc_found:
        # Try to find it in PATH
        try:
            result = subprocess.run(
                ["which", "hipcc"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  hipcc: {result.stdout.strip()} ✓")
                hipcc_found = True
        except Exception:
            pass
    
    if not hipcc_found:
        print("  hipcc: NOT FOUND (may not be needed for runtime)")
        print("    Note: hipcc is only needed for building extensions, not for inference")

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

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        has_hip = hasattr(torch.version, 'hip') and torch.version.hip
        
        if gpu_available and has_hip:
            print("✓ PyTorch ROCm/HIP support: WORKING")
            print(f"✓ GPU detected: {torch.cuda.device_count()} device(s)")
        elif gpu_available:
            print("⚠ PyTorch CUDA available but HIP version not detected")
        else:
            print("⚠ GPU not available (check device access)")
        
        try:
            import mmcv
            import mmengine
            import mmdet
            import mmdet3d
            print("✓ All MMDetection3D dependencies installed")
        except ImportError as e:
            print(f"⚠ Missing dependency: {e}")
        
        print("\n" + "=" * 80)
        print("Verification Complete")
        print("=" * 80)
        print("\nNote: GPU availability can only be verified at runtime with proper device access.")
        print("Run 'python tools/check_rocm_gpu.py' to verify GPU access and tensor operations.")
    except Exception as e:
        print(f"⚠ Could not generate summary: {e}")
        print("\n" + "=" * 80)
        print("Verification Complete")
        print("=" * 80)


if __name__ == "__main__":
    main()
