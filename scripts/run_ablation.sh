#!/bin/bash
# ==============================================================================
# AHIPS-CDR: Ablation Study Execution Script
# Verifies the independent contributions of the IPS and Dual-Level Attention.
# ==============================================================================

set -e
mkdir -p ../logs/ablation_study

# Array defining the target degradation variants for evaluation
ABLATION_MODES=("no_ips" "no_attention")

echo "========================================================================"
echo "Starting Ablation Suite for AHIPS-CDR Architectural Components..."
echo "========================================================================"

for MODE in "${ABLATION_MODES[@]}"; do
    echo "[ABLATION] Running variant profile: [${MODE}] on LinkedIn dataset..."
    
    python ../src/train.py \
        --config ../config/config.yaml \
        --dataset linkedin \
        --ablation "${MODE}" \
        > ../logs/ablation_study/linkedin_ablation_"${MODE}".log 2>&1
        
    echo "[SUCCESS] Completed variant [${MODE}]. Log saved."
    echo "------------------------------------------------------------------------"
done

echo "All ablation variants executed successfully."
