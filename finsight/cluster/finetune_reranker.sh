#!/bin/bash -l
# =============================================================================
# Fine-tune Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
# Teaches temporal relevance scoring: wrong fiscal period = not relevant
# =============================================================================

#SBATCH --job-name=finsight-rer-ft
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/finetune_reranker_%j.out
#SBATCH --error=cluster/logs/finetune_reranker_%j.err

set -e

echo "=============================================="
echo "FinSight — Reranker Fine-Tuning"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "=============================================="

cd ~/is469/finsight
mkdir -p cluster/logs models/finsight-reranker data/finetune

# ── Step 1: Ensure training data exists ──────────────────────────────────────
if [ ! -f data/finetune/reranker_train.json ]; then
    echo ""
    echo "[1/3] Generating fine-tuning training data ..."
    python3 scripts/generate_finetune_data.py
else
    echo "[1/3] Training data already exists — skipping generation"
fi

# ── Step 2: Fine-tune ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] Fine-tuning reranker ..."
python3 scripts/finetune_reranker.py \
    --epochs 4 \
    --batch-size 8 \
    --lr 2e-5 \
    --output-dir models/finsight-reranker

# ── Step 3: Summary ───────────────────────────────────────────────────────────
echo ""
echo "[3/3] Done."
EXIT=$?
echo "=============================================="
if [ $EXIT -eq 0 ]; then
    echo "✓ Reranker fine-tuning COMPLETE"
    echo "  Model: ~/is469/finsight/models/finsight-reranker/"
    echo ""
    echo "Next steps:"
    echo "  1. Update config/settings.yaml:"
    echo "       embeddings.model: models/finsight-embedder"
    echo "       reranker.model:   models/finsight-reranker"
    echo "  2. Rebuild index with fine-tuned embedder:"
    echo "       python3 scripts/build_index.py --reset"
    echo "  3. Run evaluation:"
    echo "       python3 evaluation/run_eval.py --variants v3_advanced_b --limit 20"
else
    echo "✗ Fine-tuning FAILED (exit code: $EXIT)"
fi
echo "Finished: $(date)"
echo "=============================================="
exit $EXIT
