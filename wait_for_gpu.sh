#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
TARGET_GPUS="0,1"                  # GPUs to watch
POLL_INTERVAL=1                    # seconds between checks
# ───────────────────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]"
    echo "Waits until GPUs ${TARGET_GPUS} have no running processes, then executes <command>."
    exit 1
fi

IFS=',' read -ra GPU_IDS <<< "$TARGET_GPUS"

gpu_is_free() {
    local gpu_id=$1
    # Query processes on a specific GPU; returns nothing if no compute processes
    local procs
    procs=$(nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | xargs)
    [[ -z "$procs" ]]
}

echo "🔍 Watching GPUs: ${TARGET_GPUS}"
echo "⏳ Polling every ${POLL_INTERVAL}s — waiting for all processes to finish..."
echo "🚀 Will run: $*"
echo "─────────────────────────────────────────"

while true; do
    all_free=true
    for gid in "${GPU_IDS[@]}"; do
        if ! gpu_is_free "$gid"; then
            all_free=false
            break
        fi
    done

    if $all_free; then
        echo ""
        echo "✅ $(date '+%Y-%m-%d %H:%M:%S') — GPUs ${TARGET_GPUS} are free!"
        echo "🚀 Launching: $*"
        export CUDA_VISIBLE_DEVICES="${TARGET_GPUS}"
        exec "$@"
    fi

    # Print a heartbeat every 30s so you know it's still alive
    if (( SECONDS % 30 == 0 )); then
        # Grab utilisation for a compact status line
        status=$(nvidia-smi --id="${TARGET_GPUS}" \
            --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null)
        echo "[$(date '+%H:%M:%S')] Still busy:"
        while IFS=', ' read -r idx util mem_used mem_total; do
            echo "  GPU $idx — util ${util}%  mem ${mem_used}/${mem_total} MiB"
        done <<< "$status"
    fi

    sleep "$POLL_INTERVAL"
done