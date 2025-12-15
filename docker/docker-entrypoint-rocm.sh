#!/bin/bash
# Copyright (c) OpenMMLab. All rights reserved.
# Entrypoint script for ROCm Docker container
# Runs environment verification and then starts bash

set -e

# Change to working directory (mounted at runtime)
cd /workspace/mmdetection3d 2>/dev/null || cd /mmdetection3d

# Run environment verification if script exists and SKIP_VERIFY is not set
if [ -f "tools/verify_rocm_env.py" ] && [ -z "${SKIP_VERIFY:-}" ]; then
    echo "Running environment verification..."
    echo ""
    python tools/verify_rocm_env.py
    echo ""
    echo "Environment verification complete. Starting shell..."
    echo ""
fi

# Execute the command (default: /bin/bash)
exec "$@"
