#!/usr/bin/env bash
# Run multiple simulation configurations sequentially.
# Usage: bash run_batch.sh
#
# Edit the CONFIGS array below to define your queue.
# Each line is: "flags" (passed directly to monte_carlo_sim.py)

set -e

COMMON="-n 2000"

CONFIGS=(
    "$COMMON -m light -u 500 -o results/500_light"
    "$COMMON -m light -u 2500 -o results/2500_light"
    "$COMMON -m light -u 10000 -o results/10000_light"

    "$COMMON -m default -u 500 -o results/500_default"
    "$COMMON -m default -u 2500 -o results/2500_default"
    "$COMMON -m default -u 10000 -o results/10000_default"

    "$COMMON -m heavy -u 500 -o results/500_heavy"
    "$COMMON -m heavy -u 2500 -o results/2500_heavy"
    "$COMMON -m heavy -u 10000 -o results/10000_heavy"
)

TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    echo ""
    echo "========================================"
    echo " Run $((i+1))/$TOTAL"
    echo " uv run src/simulation/monte_carlo_sim.py ${CONFIGS[$i]}"
    echo "========================================"
    echo ""
    uv run src/simulation/monte_carlo_sim.py ${CONFIGS[$i]}
done

echo ""
echo "All $TOTAL runs completed."
