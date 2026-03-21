#!/bin/bash -l
# =============================================================================
# Run Phase 1 & 2 Integration Tests on Compute Node
# This bypasses the login node's SQLite version issue
# =============================================================================

#SBATCH --job-name=phase1-2-test
#SBATCH --partition=student
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/test_%j.out
#SBATCH --error=cluster/logs/test_%j.err

set -e

echo "=============================================="
echo "Phase 1 & 2 Integration Test"
echo "Running on compute node: $(hostname)"
echo "=============================================="

# Load modules
module purge
module load python/3.11 2>/dev/null || module load python3 2>/dev/null || {
    echo "Using system Python"
}

# Navigate to project
cd ~/is469/finsight

# Check Python version
python3 --version

# Run integration test
echo ""
echo "Running integration tests..."
python3 integration_test.py

TEST_EXIT=$?

echo ""
echo "=============================================="
if [ $TEST_EXIT -eq 0 ]; then
    echo "✓ Tests PASSED"
else
    echo "✗ Tests FAILED (exit code: $TEST_EXIT)"
fi
echo "=============================================="

exit $TEST_EXIT
