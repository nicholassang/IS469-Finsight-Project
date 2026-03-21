"""
finetune_reranker.py
Fine-tunes the cross-encoder reranker on FinSight financial relevance judgments.

Goal: Teach the reranker that a passage from Q1 FY2025 is NOT relevant when the
      query asks about Q2 FY2025 — addressing temporal relevance scoring failures.

Training data: data/finetune/reranker_train.json
  Each example: {"query", "passage", "label"}   (label: 1.0 = relevant, 0.0 = not)

Loss: BinaryCrossEntropy (CrossEncoder default for regression)

Output: models/finsight-reranker/

Usage:
    python scripts/finetune_reranker.py
    python scripts/finetune_reranker.py --epochs 4 --batch-size 8
"""

import sys
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=4)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--base-model", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "models" / "finsight-reranker"))
    parser.add_argument("--train-data", type=str,
                        default=str(PROJECT_ROOT / "data" / "finetune" / "reranker_train.json"))
    parser.add_argument("--val-data",   type=str,
                        default=str(PROJECT_ROOT / "data" / "finetune" / "reranker_val.json"))
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading training data …")
    with open(args.train_data) as f:
        train_raw = json.load(f)
    with open(args.val_data) as f:
        val_raw = json.load(f)

    print(f"  Train: {len(train_raw):,} examples  "
          f"(pos={sum(1 for r in train_raw if r['label']==1.0)}, "
          f"neg={sum(1 for r in train_raw if r['label']==0.0)})")
    print(f"  Val:   {len(val_raw):,} examples")

    MAX_CHARS = 512   # cross-encoder max_length is 512 tokens; truncate text too

    # CrossEncoder expects list of [query, passage] pairs + list of labels
    train_samples = [
        [item["query"], item["passage"][:MAX_CHARS]]
        for item in train_raw
    ]
    train_labels  = [item["label"] for item in train_raw]

    val_samples = [
        [item["query"], item["passage"][:MAX_CHARS]]
        for item in val_raw
    ]
    val_labels  = [item["label"] for item in val_raw]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model …")
    model = CrossEncoder(
        args.base_model,
        num_labels=1,           # regression: output a relevance score 0–1
        max_length=512,
        device=device
    )

    # ── Evaluator ─────────────────────────────────────────────────────────────
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=val_samples,
        labels=val_labels,
        name="finsight-val"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    output_path = args.output_dir
    Path(output_path).mkdir(parents=True, exist_ok=True)

    warmup_steps = max(1, int(len(train_samples) / args.batch_size * 0.1))
    print(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, "
          f"warmup={warmup_steps}, lr={args.lr}")

    t0 = time.time()

    model.fit(
        train_dataloader=list(zip(train_samples, train_labels)),
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nRunning final evaluation …")
    score = evaluator(model)
    print(f"Final validation score: {score}")

    # ── Spot-check: does it correctly down-rank wrong-period passages? ─────────
    print("\n--- Spot-check: temporal discrimination ---")
    spot_checks = [
        {
            "query": "What was Microsoft cloud revenue in Q2 FY2025?",
            "correct_period":  "Revenue for Q2 FY2025 (quarter ended December 31, 2024) was $40.9 billion.",
            "wrong_period":    "Revenue for Q1 FY2025 (quarter ended September 30, 2024) was $38.9 billion.",
        },
        {
            "query": "What was Microsoft total revenue for FY2024?",
            "correct_period":  "Total revenue was $245,122 million for fiscal year ended June 30, 2024.",
            "wrong_period":    "Total revenue was $211,915 million for fiscal year ended June 30, 2023.",
        }
    ]
    for sc in spot_checks:
        pairs = [[sc["query"], sc["correct_period"]],
                 [sc["query"], sc["wrong_period"]]]
        scores = model.predict(pairs)
        result = "✓ PASS" if scores[0] > scores[1] else "✗ FAIL"
        print(f"\n  Query: {sc['query'][:60]}…")
        print(f"  Correct period score : {scores[0]:.4f}")
        print(f"  Wrong period score   : {scores[1]:.4f}  {result}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "base_model":       args.base_model,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "train_examples":   len(train_raw),
        "val_examples":     len(val_raw),
        "training_seconds": elapsed,
        "output_path":      output_path
    }
    with open(Path(output_path) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"""
=== Reranker fine-tuning complete ===
Model saved : {output_path}

Next steps:
  Update config/settings.yaml:
    reranker.model = {output_path}
  Then rebuild the index and run evaluation:
    python scripts/build_index.py --reset
    python evaluation/run_eval.py --variants v3_advanced_b --limit 20
""")


if __name__ == "__main__":
    main()
