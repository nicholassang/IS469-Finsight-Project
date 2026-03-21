#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh
# Full FinSight pipeline: fine-tune → rebuild index → evaluate → score RAGAS
#
# Run this on your Mac (not inside Claude's sandbox):
#   cd ~/Documents/GitHub/is469/finsight
#   bash scripts/run_full_pipeline.sh
#
# Prerequisites:
#   1. SSH tunnel to cluster is running:
#        bash cluster/tunnel.sh siken.peh.2022 10.193.104.102
#   2. vLLM job is running on cluster (check: squeue --me)
#   3. .venv is set up: python -m venv .venv && pip install -r requirements.txt
# =============================================================================

set -euo pipefail

# ── Colours ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING: $*${NC}"; }
fail() { echo -e "${RED}[$(date +%H:%M:%S)] FAIL: $*${NC}"; exit 1; }

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv/bin/python"
VLLM_URL="http://localhost:8000/v1/models"
RESULTS_DIR="$PROJECT_ROOT/evaluation/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"
log "Project root: $PROJECT_ROOT"

# ── Step 0: Activate venv ────────────────────────────────────────────────────
if [ ! -f "$VENV" ]; then
    fail ".venv not found. Run: python3 -m venv .venv && pip install -r requirements.txt"
fi
source "$PROJECT_ROOT/.venv/bin/activate"
log "Python: $(python --version)"

# ── Step 1: Component tests ──────────────────────────────────────────────────
log "===== STEP 1: Component tests ====="
python -m pytest tests/test_all_phases.py -v --tb=short \
    || fail "Component tests failed — fix before proceeding"
log "All component tests passed ✓"

# ── Step 2: Fine-tune embedding model ────────────────────────────────────────
log "===== STEP 2: Fine-tune embedding model ====="
if [ -d "$PROJECT_ROOT/models/finsight-embedder" ]; then
    warn "models/finsight-embedder already exists — skipping. Delete it to re-tune."
else
    log "Generating fine-tune data …"
    python scripts/generate_finetune_data.py

    log "Fine-tuning embedder (this takes ~10 min on CPU, ~2 min on GPU) …"
    python scripts/finetune_embedder.py --epochs 3 --batch-size 32

    log "Embedder fine-tuned ✓  →  models/finsight-embedder"
fi

# ── Step 3: Fine-tune reranker ───────────────────────────────────────────────
log "===== STEP 3: Fine-tune reranker ====="
if [ -d "$PROJECT_ROOT/models/finsight-reranker" ]; then
    warn "models/finsight-reranker already exists — skipping. Delete it to re-tune."
else
    log "Fine-tuning reranker (this takes ~5 min on CPU) …"
    python scripts/finetune_reranker.py --epochs 4 --batch-size 8

    log "Reranker fine-tuned ✓  →  models/finsight-reranker"
fi

# ── Step 4: Update config to use fine-tuned models ───────────────────────────
log "===== STEP 4: Activating fine-tuned models in config ====="
python - <<'PYEOF'
import re, sys
from pathlib import Path

cfg_path = Path("config/settings.yaml")
txt = cfg_path.read_text()

# Swap embedder
txt = re.sub(
    r"(embeddings:\s*\n(?:.*\n)*?.*?)model:\s*sentence-transformers/all-mpnet-base-v2",
    r"\1model: models/finsight-embedder",
    txt, count=1
)
# Swap reranker
txt = re.sub(
    r"(reranker:\s*\n(?:.*\n)*?.*?)model:\s*cross-encoder/ms-marco-MiniLM-L-6-v2",
    r"\1model: models/finsight-reranker",
    txt, count=1
)
cfg_path.write_text(txt)
print("config/settings.yaml updated to use fine-tuned models")
PYEOF

# ── Step 5: Re-ingest with semantic chunking + rebuild ChromaDB index ─────────
log "===== STEP 5: Re-ingesting with semantic chunking (experiment_D) ====="
log "This re-chunks all 9 documents using SEC-aware + table-preserving splitting …"

python scripts/ingest_all.py --chunking experiment_D \
    || fail "ingest_all.py failed"

log "Ingestion complete ✓  (data/processed/*.json updated with semantic chunks)"

log "===== STEP 5b: Rebuilding ChromaDB index (fine-tuned embedder) ====="
python scripts/build_index.py --reset \
    || fail "build_index.py failed"

log "Index rebuilt ✓  (semantic chunks + fine-tuned embedder)"

# ── Step 6: Check vLLM is reachable ──────────────────────────────────────────
log "===== STEP 6: Checking vLLM tunnel ====="
if ! curl -s --max-time 5 "$VLLM_URL" > /dev/null 2>&1; then
    fail "vLLM not reachable at $VLLM_URL.
  → Start tunnel:  bash cluster/tunnel.sh siken.peh.2022 10.193.104.102
  → Check job:     ssh siken.peh.2022@10.193.104.102 'squeue --me'"
fi
MODEL=$(curl -s "$VLLM_URL" | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
log "vLLM is live — model: $MODEL ✓"

# ── Step 7: Run evaluation (answer generation) ───────────────────────────────
log "===== STEP 7: Running evaluation (v3_advanced_b, 20 questions) ====="
EVAL_OUT="$RESULTS_DIR/fresh_${TIMESTAMP}"
mkdir -p "$EVAL_OUT"

python evaluation/run_eval.py \
    --variants v3_advanced_b \
    --output "$EVAL_OUT" \
    || fail "run_eval.py failed"

RESULT_FILE="$EVAL_OUT/v3_advanced_b.json"
log "Answers generated ✓  →  $RESULT_FILE"

# ── Step 8: RAGAS scoring ────────────────────────────────────────────────────
log "===== STEP 8: RAGAS scoring ====="
SCORED_FILE="$EVAL_OUT/v3_advanced_b_scored.json"

python evaluation/rescore_ragas.py \
    --input  "$RESULT_FILE" \
    --output "$SCORED_FILE" \
    || fail "rescore_ragas.py failed"

log "RAGAS scoring complete ✓  →  $SCORED_FILE"

# ── Step 9: Print comparison ──────────────────────────────────────────────────
log "===== STEP 9: Results comparison ====="
python - "$SCORED_FILE" <<'PYEOF'
import sys, json

with open(sys.argv[1]) as f:
    new = json.load(f)

# Try to get aggregate from new results
new_agg = {}
if isinstance(new, dict):
    for variant, vdata in new.items():
        if isinstance(vdata, dict) and "aggregate" in vdata:
            new_agg = vdata["aggregate"]
            break

# Stored baseline and old advanced results for comparison
baseline = {"faithfulness": 0.607,  "answer_relevancy": 0.4849, "context_recall": 0.225,  "context_precision": 0.4555}
old_adv  = {"faithfulness": 0.8393, "answer_relevancy": 0.9042, "context_recall": 0.7333, "context_precision": 0.5263}

metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
labels  = ["Faithfulness", "Ans Relevancy",    "Context Recall", "Ctx Precision"]

print()
print("=" * 75)
print("  FinSight RAGAS Results — After Fine-tuning + Semantic Chunking")
print("=" * 75)
print(f"  {'Metric':<22} {'Baseline':>10} {'Pre-finetune':>14} {'Post-finetune':>15}")
print("  " + "─" * 62)
for m, l in zip(metrics, labels):
    b = baseline.get(m, 0.0)
    o = old_adv.get(m, 0.0)
    n = new_agg.get(m, float("nan"))
    delta = n - b if n == n else float("nan")
    sign = "+" if delta > 0 else ""
    print(f"  {l:<22} {b:>10.4f} {o:>14.4f} {n:>14.4f}  ({sign}{delta:.4f} vs baseline)")
print("=" * 75)
print()
PYEOF

log "===== PIPELINE COMPLETE ====="
log "All results in: $RESULTS_DIR/fresh_${TIMESTAMP}/"
