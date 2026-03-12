#!/usr/bin/env bash
# Run multiple simulation configurations sequentially.
# Usage: bash run_batch.sh
#
# Edit the CONFIGS array below to define your queue.
# Each line is: "flags" (passed directly to monte_carlo_sim.py)

set -e

COMMON="-n 2000 -s 8 12"

CONFIGS=(
    "$COMMON -m v_heavy -u 10000 -o results/10000_heavy_2"
)

TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    echo ""
    echo "========================================"
    echo " Run $((i+1))/$TOTAL"
    echo " uv run src/monte_carlo_sim.py ${CONFIGS[$i]}"
    echo "========================================"
    echo ""
    uv run src/monte_carlo_sim.py ${CONFIGS[$i]}
done

echo ""
echo "All $TOTAL runs completed."
