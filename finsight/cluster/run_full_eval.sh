#!/bin/bash -l
# =============================================================================
# run_full_eval.sh — Full pipeline on cluster: finetune → ingest → eval
#
# Submit with:
#   sbatch cluster/run_full_eval.sh
#
# Or run steps individually:
#   sbatch cluster/finetune_embedder.sh
#   sbatch --dependency=afterok:<emb_job_id> cluster/finetune_reranker.sh
#   sbatch --dependency=afterok:<rer_job_id> cluster/run_eval_only.sh
# =============================================================================

#SBATCH --job-name=finsight-full-eval
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=cluster/logs/full_eval_%j.out
#SBATCH --error=cluster/logs/full_eval_%j.err

set -e

echo "=============================================="
echo "FinSight Full Pipeline — Cluster Job"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "=============================================="

cd ~/is469/finsight
mkdir -p cluster/logs models data/finetune evaluation/results

# ── 1. Generate fine-tune data ────────────────────────────────────────────────
echo ""
echo "[1/6] Generating fine-tuning data ..."
if [ ! -f data/finetune/embedder_train.json ]; then
    python3 scripts/generate_finetune_data.py
else
    echo "  Fine-tune data already exists — skipping"
fi

# ── 2. Fine-tune embedder ─────────────────────────────────────────────────────
echo ""
echo "[2/6] Fine-tuning embedding model ..."
if [ ! -d models/finsight-embedder ]; then
    python3 scripts/finetune_embedder.py \
        --epochs 3 --batch-size 16 --lr 2e-5 \
        --output-dir models/finsight-embedder
    echo "  Embedder fine-tuned ✓"
else
    echo "  models/finsight-embedder exists — skipping"
fi

# ── 3. Fine-tune reranker ─────────────────────────────────────────────────────
echo ""
echo "[3/6] Fine-tuning reranker ..."
if [ ! -d models/finsight-reranker ]; then
    python3 scripts/finetune_reranker.py \
        --epochs 4 --batch-size 8 --lr 2e-5 \
        --output-dir models/finsight-reranker
    echo "  Reranker fine-tuned ✓"
else
    echo "  models/finsight-reranker exists — skipping"
fi

# ── 4. Update config to use fine-tuned models ─────────────────────────────────
echo ""
echo "[4/6] Updating config/settings.yaml to use fine-tuned models ..."
python3 - <<'PYEOF'
import re
from pathlib import Path

cfg_path = Path("config/settings.yaml")
txt = cfg_path.read_text()

# Swap embedder (only if still pointing at HuggingFace model)
if "sentence-transformers/all-mpnet-base-v2" in txt:
    txt = re.sub(
        r"(embeddings:.*?\n(?:.*?\n)*?.*?)model:\s*sentence-transformers/all-mpnet-base-v2",
        r"\1model: models/finsight-embedder",
        txt, count=1, flags=re.DOTALL
    )
    print("  Embedder model → models/finsight-embedder")

# Swap reranker (only if still pointing at HuggingFace model)
if "cross-encoder/ms-marco-MiniLM-L-6-v2" in txt:
    txt = re.sub(
        r"(reranker:.*?\n(?:.*?\n)*?.*?)model:\s*cross-encoder/ms-marco-MiniLM-L-6-v2",
        r"\1model: models/finsight-reranker",
        txt, count=1, flags=re.DOTALL
    )
    print("  Reranker model → models/finsight-reranker")

cfg_path.write_text(txt)
print("  config/settings.yaml updated")
PYEOF

# ── 5. Re-ingest with semantic chunking + rebuild index ───────────────────────
echo ""
echo "[5/6] Re-ingesting with experiment_D (semantic chunking) + rebuilding index ..."
python3 scripts/ingest_all.py --chunking experiment_D
python3 scripts/build_index.py --reset
echo "  Index rebuilt ✓"

# ── 6. Run evaluation (requires vLLM running on this cluster) ─────────────────
echo ""
echo "[6/6] Running evaluation against vLLM ..."

# Check if vLLM is reachable on localhost:8000
VLLM_READY=false
for i in {1..5}; do
    if curl -s --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1; then
        VLLM_READY=true
        break
    fi
    echo "  Waiting for vLLM to be ready (attempt $i/5) ..."
    sleep 15
done

if [ "$VLLM_READY" = false ]; then
    echo "  WARNING: vLLM not reachable on localhost:8000"
    echo "  Ensure finsight-vllm job is running: squeue --me"
    echo "  Skipping generation step — run manually after vLLM starts:"
    echo "    python3 evaluation/run_eval.py --variants v3_advanced_b"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_OUT="evaluation/results/fresh_${TIMESTAMP}"
mkdir -p "$EVAL_OUT"

python3 evaluation/run_eval.py \
    --variants v3_advanced_b \
    --output "$EVAL_OUT"

echo ""
echo "[6/6b] RAGAS scoring ..."
python3 evaluation/rescore_ragas.py \
    --input  "$EVAL_OUT/v3_advanced_b.json" \
    --output "$EVAL_OUT/v3_advanced_b_scored.json"

# ── Print summary ──────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="
python3 - "$EVAL_OUT/v3_advanced_b_scored.json" <<'PYEOF'
import sys, json, math

with open(sys.argv[1]) as f:
    data = json.load(f)

agg = {}
for k, v in data.items():
    if isinstance(v, dict) and "aggregate" in v:
        agg = v["aggregate"]
        break

baseline = {
    "faithfulness": 0.607, "answer_relevancy": 0.4849,
    "context_recall": 0.225, "context_precision": 0.4555
}

metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
labels  = ["Faithfulness", "Ans Relevancy", "Context Recall", "Ctx Precision"]

print(f"  {'Metric':<22} {'Baseline':>10} {'New':>10} {'Delta':>10}")
print("  " + "-" * 55)
for m, l in zip(metrics, labels):
    b = baseline.get(m, 0.0)
    n = agg.get(m, math.nan)
    d = n - b if not math.isnan(n) else math.nan
    sign = "+" if d > 0 else ""
    print(f"  {l:<22} {b:>10.4f} {n:>10.4f} {sign}{d:>9.4f}")
PYEOF

echo ""
echo "Full results: $EVAL_OUT/"
echo "Finished: $(date)"
echo "=============================================="
