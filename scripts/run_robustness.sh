#!/bin/bash
# ==============================================================================
# AHIPS-CDR: Progressive Data Sparsity & Robustness Evaluation Script
# Simulates interaction dropouts by artificially removing 0% to 40% of training edges.
# ==============================================================================

set -e
mkdir -p ../logs/robustness_analysis

# Define progressive sparsity perturbation constraints (0% to 40% edge removal)
SPARSITY_LEVELS=(0.0 0.1 0.2 0.3 0.4)

echo "========================================================================"
echo "Commencing Robustness Profiling under Artificially Induced Data Sparsity..."
echo "========================================================================"

for SPARSITY in "${SPARSITY_LEVELS[@]}"; do
    # Convert decimal representation to clean percentage string for filenames
    PCT=$(echo "$SPARSITY * 100" | bc | cut -d'.' -f1)
    echo "[ROBUSTNESS] Simulating ${PCT}% interaction edge dropout on LinkedIn..."
    
    python ../src/train.py \
        --config ../config/config.yaml \
        --dataset linkedin \
        --ablation none \
        --sparsity_ratio "${SPARSITY}" \
        > ../logs/robustness_analysis/linkedin_sparsity_${PCT}pct.log 2>&1
        
    echo "[SUCCESS] Finished training at ${PCT}% sparsity level."
    echo "------------------------------------------------------------------------"
done

echo "Data sparsity robustness matrix generation completed successfully."
