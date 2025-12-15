#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Quick ROCm / AMD GPU accessibility check for the Docker image.

Run this inside the ROCm-based container (e.g. built from
`docker/Dockerfile.rocm`) to verify that:

- PyTorch can see at least one CUDA/HIP device.
- Basic tensor operations on the GPU succeed.
"""
import sys
import time

import torch


def main() -> int:
    print('=== ROCm / CUDA device visibility ===')
    print(f'PyTorch version: {torch.__version__}')
    print(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")

    cuda_available = torch.cuda.is_available()
    print(f'torch.cuda.is_available(): {cuda_available}')

    if not cuda_available:
        print(
            'No CUDA/HIP devices visible. '
            'Check that /dev/kfd and /dev/dri are passed to the container '
            'and that the user has permissions to access them.',
            file=sys.stderr,
        )
        return 1

    num_devices = torch.cuda.device_count()
    print(f'Number of visible devices: {num_devices}')
    for idx in range(num_devices):
        name = torch.cuda.get_device_name(idx)
        print(f'  [{idx}] {name}')

    print('\n=== Simple tensor test on cuda:0 ===')
    device = torch.device('cuda:0')
    try:
        a = torch.randn((1024, 1024), device=device)
        b = torch.randn((1024, 1024), device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        c = a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0
        print(f'Matrix multiply (1024x1024) took {dt:.2f} ms on {device}')
        print(f'Result tensor shape: {tuple(c.shape)}')
    except Exception as exc:  # noqa: BLE001
        print('Error while running tensor operations on cuda:0:', exc)
        return 1

    print('\nROCm GPU check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


