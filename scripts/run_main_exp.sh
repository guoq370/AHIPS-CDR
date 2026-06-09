#!/bin/bash
# ==============================================================================
# AHIPS-CDR: Unified Main Execution Script (Updated Version)
# Fully executes the proposed model and all major state-of-the-art baselines.
# ==============================================================================

set -e
mkdir -p ../logs/main_experiments

echo "========================================================================"
echo "Phase 1: Running Proposed Model (AHIPS-CDR)"
echo "========================================================================"
python ../src/train.py --dataset linkedin --ablation none > ../logs/main_experiments/ahips_linkedin.log 2>&1
echo "[SUCCESS] AHIPS-CDR processing finalized."

echo ""
echo "========================================================================"
echo "Phase 2: Running Comprehensive Comparative Baselines Suite"
echo "========================================================================"

# Array containing baseline target keywords
TARGET_BASELINES=("HGT" "HAGO" "MI-DTCDR" "SEAGULL")

for BASELINE in "${TARGET_BASELINES[@]}"; do
    echo "[LAUNCHING BASELINE] Executing ${BASELINE} pipeline pass..."
    python ../baselines/baselines_runner.py --model "${BASELINE}" >> ../logs/main_experiments/baselines_comparison.log 2>&1
done

echo "[SUCCESS] Main experiments and baseline benchmarks completed perfectly."
echo "Check '../logs/main_experiments/baselines_comparison.log' for unified metrics data."
echo "========================================================================"
