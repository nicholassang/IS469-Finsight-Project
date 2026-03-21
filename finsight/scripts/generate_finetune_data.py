"""
generate_finetune_data.py
Builds training data for fine-tuning the embedding model and reranker.

Sources:
  - evaluation/eval_dataset.json          (20 Q&A pairs with source_doc labels)
  - data/processed/*.json                 (1,423 chunks with fiscal_period metadata)
  - evaluation/results/eval_results_improved_v3.json  (retrieved contexts per query)

Outputs:
  - data/finetune/embedder_train.json     (query, positive, hard_negative triplets)
  - data/finetune/reranker_train.json     (query, passage, label triplets)

Strategy:
  Positive pairs  : question + chunks from the labelled source document that
                    overlap with keywords in the ground-truth answer.
  Hard negatives  : chunks from a DIFFERENT fiscal period that share topic
                    keywords — these are the exact failure mode we want to fix.
  Soft negatives  : random chunks from unrelated documents.
"""

import json
import os
import re
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH    = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "eval_results_improved_v3.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR      = PROJECT_ROOT / "data" / "finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_keywords(text: str, min_len: int = 4) -> set:
    """Return lower-cased alpha-numeric words longer than min_len."""
    return {w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{%d,}\b' % min_len, text)}


def keyword_overlap(text: str, keywords: set) -> float:
    """Fraction of keywords that appear in text."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


def source_doc_to_fiscal_period(source_doc: str) -> str:
    """
    Map source_doc id → fiscal_period string used in chunk metadata.
    e.g. 'msft_10k_fy2024'    → 'FY2024'
         'msft_10q_q2_fy2025' → 'Q2 FY2025'
    """
    s = source_doc.lower()
    # quarterly: msft_10q_q2_fy2025
    m = re.search(r'10q_q(\d)_fy(\d{4})', s)
    if m:
        return f"Q{m.group(1)} FY{m.group(2)}"
    # annual: msft_10k_fy2024
    m = re.search(r'fy(\d{4})', s)
    if m:
        return f"FY{m.group(1)}"
    return ""

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading evaluation dataset …")
with open(EVAL_PATH) as f:
    eval_data = json.load(f)

print("Loading eval results …")
with open(RESULTS_PATH) as f:
    results_raw = json.load(f)
results_by_id = {r["id"]: r for r in results_raw["advanced"]["per_question"]}

print("Loading processed chunks …")
all_chunks = []
chunks_by_doc   = defaultdict(list)   # source_file stem → chunks
chunks_by_period = defaultdict(list)  # fiscal_period → chunks

for json_file in sorted(PROCESSED_DIR.glob("*.json")):
    with open(json_file) as f:
        doc_chunks = json.load(f)
    for c in doc_chunks:
        all_chunks.append(c)
        doc_stem = Path(c["metadata"]["source_file"]).stem   # e.g. msft_10k_fy2024
        chunks_by_doc[doc_stem].append(c)
        fp = c["metadata"]["fiscal_period"]                  # e.g. FY2024
        chunks_by_period[fp].append(c)

print(f"  Total chunks: {len(all_chunks)}")
print(f"  Documents:    {len(chunks_by_doc)}")
print(f"  Periods:      {sorted(chunks_by_period.keys())}")

# ── Build training examples ───────────────────────────────────────────────────

embedder_examples = []   # {"query", "positive", "hard_negative"}
reranker_examples = []   # {"query", "passage", "label"}

N_POSITIVES     = 5   # positive chunks per question
N_HARD_NEG      = 5   # hard negatives (wrong period, same topic) per question
N_SOFT_NEG      = 3   # random negatives per question
KEYWORD_THRESH  = 0.08  # min keyword overlap to count as positive

all_periods = list(chunks_by_period.keys())

for item in eval_data:
    qid        = item["id"]
    question   = item["question"]
    ground_truth = item["ground_truth"]
    source_doc = item["source_doc"]                         # e.g. msft_10k_fy2024
    target_period = source_doc_to_fiscal_period(source_doc) # e.g. FY2024

    gt_keywords = extract_keywords(ground_truth)
    q_keywords  = extract_keywords(question)
    all_keywords = gt_keywords | q_keywords

    # ── Positives: high-overlap chunks from the labelled source doc ──────────
    source_chunks = chunks_by_doc.get(source_doc, [])
    scored = [(keyword_overlap(c["text"], all_keywords), c) for c in source_chunks]
    scored.sort(key=lambda x: -x[0])

    positives = [c for score, c in scored if score >= KEYWORD_THRESH][:N_POSITIVES]
    if not positives:
        # Fall back: take top-scored chunks regardless of threshold
        positives = [c for _, c in scored[:3]]

    # Also include contexts from eval results as positives (they came from the
    # retriever for this exact question so are high-quality signal)
    eval_contexts = results_by_id.get(qid, {}).get("contexts", [])
    extra_positives = [{"text": ctx, "metadata": {"fiscal_period": target_period}}
                       for ctx in eval_contexts[:3]]
    positives = positives + extra_positives

    # ── Hard negatives: same topic keywords, DIFFERENT fiscal period ─────────
    other_periods = [p for p in all_periods if p != target_period]
    hard_neg_pool = []
    for other_period in other_periods:
        period_chunks = chunks_by_period[other_period]
        scored_neg = [(keyword_overlap(c["text"], q_keywords), c)
                      for c in period_chunks]
        scored_neg.sort(key=lambda x: -x[0])
        # Only include chunks that share topic keywords (non-zero overlap)
        hard_neg_pool.extend([c for sc, c in scored_neg if sc > 0][:2])

    random.shuffle(hard_neg_pool)
    hard_negatives = hard_neg_pool[:N_HARD_NEG]

    # ── Soft negatives: random unrelated chunks ───────────────────────────────
    unrelated = [c for c in all_chunks
                 if c["metadata"]["fiscal_period"] != target_period
                 and keyword_overlap(c["text"], q_keywords) == 0]
    soft_negatives = random.sample(unrelated, min(N_SOFT_NEG, len(unrelated)))

    # ── Write embedder examples ───────────────────────────────────────────────
    for pos in positives:
        pos_text = pos["text"] if isinstance(pos, dict) else pos
        # Pair with each hard negative as a triplet
        for neg in hard_negatives:
            neg_text = neg["text"] if isinstance(neg, dict) else neg
            embedder_examples.append({
                "query":         question,
                "positive":      pos_text,
                "hard_negative": neg_text,
                "qid":           qid,
                "source_period": target_period,
                "neg_period":    neg["metadata"]["fiscal_period"] if isinstance(neg, dict) else "unknown"
            })
        # Also add soft negatives
        for neg in soft_negatives:
            embedder_examples.append({
                "query":         question,
                "positive":      pos_text,
                "hard_negative": neg["text"],
                "qid":           qid,
                "source_period": target_period,
                "neg_period":    neg["metadata"]["fiscal_period"]
            })

    # ── Write reranker examples ───────────────────────────────────────────────
    for pos in positives:
        pos_text = pos["text"] if isinstance(pos, dict) else pos
        reranker_examples.append({
            "query":   question,
            "passage": pos_text,
            "label":   1.0,
            "qid":     qid,
            "period":  target_period
        })
    for neg in hard_negatives:
        neg_text = neg["text"] if isinstance(neg, dict) else neg
        reranker_examples.append({
            "query":   question,
            "passage": neg_text,
            "label":   0.0,    # wrong period = irrelevant
            "qid":     qid,
            "period":  neg["metadata"]["fiscal_period"] if isinstance(neg, dict) else "unknown"
        })
    for neg in soft_negatives:
        reranker_examples.append({
            "query":   question,
            "passage": neg["text"],
            "label":   0.0,
            "qid":     qid,
            "period":  neg["metadata"]["fiscal_period"]
        })

# ── Shuffle and split ─────────────────────────────────────────────────────────
random.shuffle(embedder_examples)
random.shuffle(reranker_examples)

# Hold out last 10% as validation
def split(data, val_frac=0.1):
    n_val = max(1, int(len(data) * val_frac))
    return data[n_val:], data[:n_val]

emb_train, emb_val = split(embedder_examples)
rer_train, rer_val = split(reranker_examples)

# ── Save ──────────────────────────────────────────────────────────────────────
def save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data):,} examples → {path}")

print("\nSaving training data …")
save(emb_train, OUT_DIR / "embedder_train.json")
save(emb_val,   OUT_DIR / "embedder_val.json")
save(rer_train, OUT_DIR / "reranker_train.json")
save(rer_val,   OUT_DIR / "reranker_val.json")

print(f"""
=== Fine-tune data generation complete ===
Embedder  : {len(emb_train)} train / {len(emb_val)} val triplets
Reranker  : {len(rer_train)} train / {len(rer_val)} val examples
  Labels  : {sum(1 for r in rer_train if r['label']==1.0)} positive
             {sum(1 for r in rer_train if r['label']==0.0)} negative
Output dir: {OUT_DIR}
Next steps:
  python scripts/finetune_embedder.py
  python scripts/finetune_reranker.py
""")
