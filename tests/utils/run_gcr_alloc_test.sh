#!/bin/bash
# Test whether GCR tracks PyTorch allocations under different allocator configs.
#
# Usage:
#   bash tests/utils/run_gcr_alloc_test.sh
#
# Set GCR_HOME if not already set (default: /root/GCR)

set -euo pipefail


export GCR_HOME="${GCR_HOME:-/root/GCR}"
PRELOAD="$GCR_HOME/GCR/libpreload.so:$GCR_HOME/GCR/libcuda.so"
SCRIPT="$(dirname "$0")/test_gcr_alloc_tracking.py"

run_test() {
    local label="$1"
    local alloc_conf="$2"

    echo "============================================"
    echo "  $label"
    echo "  PYTORCH_ALLOC_CONF=$alloc_conf"
    echo "============================================"

    PYTORCH_ALLOC_CONF="$alloc_conf" \
    LD_PRELOAD="$PRELOAD" \
        python "$SCRIPT"

    echo ""
}

run_test "1. Default (expandable_segments on)" \
        "expandable_segments:True"

run_test "2. expandable_segments OFF (cudaMalloc path)" \
        "expandable_segments:False"

run_test "3. cudaMallocAsync backend" \
        "backend:cudaMallocAsync"

echo "Done. Compare gcr_va across the three runs."
