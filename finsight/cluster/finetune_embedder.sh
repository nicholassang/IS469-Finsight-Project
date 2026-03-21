#!/bin/bash -l
# =============================================================================
# Fine-tune Embedding Model (sentence-transformers/all-mpnet-base-v2)
# Teaches fiscal period discrimination: Q1 FY2025 ≠ Q2 FY2025
# =============================================================================

#SBATCH --job-name=finsight-emb-ft
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/finetune_embedder_%j.out
#SBATCH --error=cluster/logs/finetune_embedder_%j.err

set -e

echo "=============================================="
echo "FinSight — Embedding Model Fine-Tuning"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "=============================================="

cd ~/is469/finsight
mkdir -p cluster/logs models/finsight-embedder data/finetune

# ── Step 1: Generate training data (if not already done) ─────────────────────
if [ ! -f data/finetune/embedder_train.json ]; then
    echo ""
    echo "[1/3] Generating fine-tuning training data ..."
    python3 scripts/generate_finetune_data.py
else
    echo "[1/3] Training data already exists — skipping generation"
    echo "      (delete data/finetune/ and re-run to regenerate)"
fi

# ── Step 2: Fine-tune ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] Fine-tuning embedding model ..."
python3 scripts/finetune_embedder.py \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --output-dir models/finsight-embedder

# ── Step 3: Summary ───────────────────────────────────────────────────────────
echo ""
echo "[3/3] Done."
EXIT=$?
echo "=============================================="
if [ $EXIT -eq 0 ]; then
    echo "✓ Embedding fine-tuning COMPLETE"
    echo "  Model: ~/is469/finsight/models/finsight-embedder/"
    echo ""
    echo "Next: sbatch cluster/finetune_reranker.sh"
else
    echo "✗ Fine-tuning FAILED (exit code: $EXIT)"
fi
echo "Finished: $(date)"
echo "=============================================="
exit $EXIT
