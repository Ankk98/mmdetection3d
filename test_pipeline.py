#!/usr/bin/env python3
"""
Simple test to verify the SiT dataset pipeline works
"""

import torch
import numpy as np
from pathlib import Path

def test_data_loading():
    """Test basic data loading without full MMDet3D pipeline"""
    print("Testing SiT dataset loading...")

    # Check if data exists
    data_root = Path("data/sit_test")
    if not data_root.exists():
        print("❌ Data directory not found")
        return False

    # Check converted files
    bin_files = list((data_root / "training" / "velodyne").glob("*.bin"))
    print(f"Found {len(bin_files)} .bin files")

    if len(bin_files) == 0:
        print("❌ No .bin files found")
        return False

    # Load a sample point cloud
    sample_file = bin_files[0]
    points = np.fromfile(str(sample_file), dtype=np.float32).reshape(-1, 4)
    print(f"✓ Loaded {len(points)} points from {sample_file.name}")
    print(f"  Point cloud shape: {points.shape}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    # Check info files
    import mmengine
    info_file = data_root / "sit_infos_train.pkl"
    if info_file.exists():
        infos = mmengine.load(info_file)
        print(f"✓ Loaded info file with {len(infos['data_list'])} samples")
        print(f"  Classes: {infos['metainfo']['categories']}")
    else:
        print("❌ Info file not found")
        return False

    print("✓ Basic data loading test passed!")
    return True

def test_imports():
    """Test basic imports work"""
    print("Testing basic imports...")

    try:
        import torch
        print(f"✓ PyTorch version: {torch.version.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  ROCm available: {torch.version.hip is not None}")

        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")

        import open3d as o3d
        print(f"✓ Open3D version: {o3d.__version__}")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SiT Dataset Pipeline Test")
    print("=" * 50)

    success = True
    success &= test_imports()
    success &= test_data_loading()

    print("=" * 50)
    if success:
        print("🎉 All tests passed! Pipeline is ready.")
    else:
        print("❌ Some tests failed. Check the issues above.")
    print("=" * 50)
