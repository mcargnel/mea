#!/usr/bin/env bash
# Run multiple simulation configurations sequentially.
# Usage: bash run_batch.sh
#
# Edit the CONFIGS array below to define your queue.
# Each line is: "flags" (passed directly to monte_carlo_sim.py)

set -e

COMMON="-n 300 -s 1 2 3 4 5 6 7 8 9 10 11 12"

CONFIGS=(
    "$COMMON -m light -u 500 -o results/500_light"
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
