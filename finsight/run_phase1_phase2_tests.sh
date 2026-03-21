#!/bin/bash
# =============================================================================
# run_phase1_phase2_tests.sh
# Test runner for Phase 1 & Phase 2 improvements
#
# Usage:
#   bash run_phase1_phase2_tests.sh
# =============================================================================

set -e

echo "=============================================="
echo "Phase 1 & Phase 2 Test Runner"
echo "=============================================="

# ── Load modules ──────────────────────────────────────────────────────────────
echo "[1/3] Loading Python environment..."
module load python/3.11 2>/dev/null || module load python3 2>/dev/null || {
    echo "WARNING: Could not load python module."
}

# Activate venv
if [ -d ~/finsight-llm/venv ]; then
    source ~/finsight-llm/venv/bin/activate
    echo "Using virtual environment: ~/finsight-llm/venv"
else
    echo "WARNING: Virtual environment not found. Using system Python."
fi

# ── Run tests ─────────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Running component tests..."
cd ~/is469/finsight
python test_phase1_phase2.py

TEST_EXIT=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Test summary"
if [ $TEST_EXIT -eq 0 ]; then
    echo "✓ All tests passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Run full evaluation:"
    echo "     python evaluation/run_eval.py --variants v3_advanced_b --limit 20"
    echo ""
    echo "  2. Check specific failure cases:"
    echo "     # q007: Temporal accuracy"
    echo "     # q015: Token limits"
    echo "     # q010: Insufficient evidence"
else
    echo "✗ Tests failed with exit code $TEST_EXIT"
    echo "Check PHASE1_PHASE2_STATUS.md for troubleshooting"
fi

echo "=============================================="
exit $TEST_EXIT
