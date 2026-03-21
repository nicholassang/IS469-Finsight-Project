"""
finetune_embedder.py
Fine-tunes the sentence-transformer embedding model on FinSight financial data.

Goal: Teach the model that "Q2 FY2025" and "Q1 FY2025" are semantically DIFFERENT
      even though they share most tokens — eliminating the temporal confusion that
      caused q007-style failures.

Training data: data/finetune/embedder_train.json
  Each example: {"query", "positive", "hard_negative"}

Loss: TripletLoss with hard negatives
  - Pulls (query, positive) together
  - Pushes (query, hard_negative) apart — especially wrong-period chunks

Output: models/finsight-embedder/

Usage:
    python scripts/finetune_embedder.py
    python scripts/finetune_embedder.py --epochs 5 --batch-size 16
"""

import sys
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  (patches sqlite3 if needed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--warmup",     type=int,   default=100)
    parser.add_argument("--base-model", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "models" / "finsight-embedder"))
    parser.add_argument("--train-data", type=str,
                        default=str(PROJECT_ROOT / "data" / "finetune" / "embedder_train.json"))
    parser.add_argument("--val-data",   type=str,
                        default=str(PROJECT_ROOT / "data" / "finetune" / "embedder_val.json"))
    args = parser.parse_args()

    # ── Imports (heavy, import after arg parse so --help is fast) ─────────────
    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from torch.utils.data import DataLoader
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.output_dir}")

    # ── Load training data ────────────────────────────────────────────────────
    print(f"\nLoading training data from {args.train_data} …")
    with open(args.train_data) as f:
        train_raw = json.load(f)

    print(f"Loading validation data from {args.val_data} …")
    with open(args.val_data) as f:
        val_raw = json.load(f)

    print(f"  Train: {len(train_raw):,} triplets")
    print(f"  Val:   {len(val_raw):,} triplets")

    # ── Build InputExamples ───────────────────────────────────────────────────
    # TripletLoss expects: anchor, positive, negative
    MAX_CHARS = 1000   # truncate long passages to keep GPU memory sane

    def make_examples(data):
        return [
            InputExample(texts=[
                item["query"],
                item["positive"][:MAX_CHARS],
                item["hard_negative"][:MAX_CHARS]
            ])
            for item in data
        ]

    train_examples = make_examples(train_raw)
    val_examples   = make_examples(val_raw)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model …")
    model = SentenceTransformer(args.base_model, device=device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size
    )

    # ── Loss: TripletLoss pushes hard negatives (wrong fiscal period) away ────
    train_loss = losses.TripletLoss(model=model)

    # ── Evaluator: cosine similarity on val set ───────────────────────────────
    # Build anchor/positive/negative sentence lists for the evaluator
    anchors   = [ex.texts[0] for ex in val_examples]
    positives = [ex.texts[1] for ex in val_examples]
    negatives = [ex.texts[2] for ex in val_examples]

    evaluator = evaluation.TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="finsight-val",
        show_progress_bar=False
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    output_path = args.output_dir
    Path(output_path).mkdir(parents=True, exist_ok=True)

    total_steps = len(train_loader) * args.epochs
    print(f"\nTraining for {args.epochs} epochs × {len(train_loader)} steps = {total_steps} total steps")
    print(f"Warmup steps: {args.warmup}")
    print(f"Learning rate: {args.lr}")
    print()

    t0 = time.time()

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup,
        optimizer_params={"lr": args.lr},
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Fine-tuned model saved to: {output_path}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nRunning final evaluation on validation set …")
    score = evaluator(model)
    print(f"Final triplet accuracy: {score:.4f}")

    # ── Save training metadata ────────────────────────────────────────────────
    meta = {
        "base_model":      args.base_model,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "train_examples":  len(train_raw),
        "val_examples":    len(val_raw),
        "final_val_score": score,
        "training_seconds": elapsed,
        "output_path":     output_path
    }
    with open(Path(output_path) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved.")

    print(f"""
=== Embedding fine-tuning complete ===
Model saved : {output_path}
Val accuracy: {score:.4f}  (higher = model better separates wrong-period chunks)

Next step:
  python scripts/finetune_reranker.py
  → then update config/settings.yaml: embeddings.model = {output_path}
""")


if __name__ == "__main__":
    main()
