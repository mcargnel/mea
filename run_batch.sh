#!/usr/bin/env bash
# Run all 9 simulation configurations: {500,2500,10000} x {light,default,heavy}.
# Usage: bash run_batch.sh

set -euo pipefail

COMMON="-n 2000"
PRESETS=(light default heavy)
N_UNITS=(500 2500 10000)

TOTAL=$(( ${#PRESETS[@]} * ${#N_UNITS[@]} ))
i=0
for preset in "${PRESETS[@]}"; do
    for n in "${N_UNITS[@]}"; do
        i=$((i+1))
        out="output/simulations/${n}_${preset}"
        echo
        echo "========================================"
        echo " Run $i/$TOTAL  -m $preset -u $n -> $out"
        echo "========================================"
        echo
        uv run src/simulation/monte_carlo_sim.py $COMMON -m "$preset" -u "$n" -o "$out"
    done
done

echo
echo "All $TOTAL runs completed."
