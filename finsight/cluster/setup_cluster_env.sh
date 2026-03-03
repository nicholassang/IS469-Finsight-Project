#!/bin/bash
# =============================================================================
# FinSight — One-time cluster environment setup
# Run this ONCE on the login node before submitting any SLURM jobs.
#
# Usage:
#   chmod +x cluster/setup_cluster_env.sh
#   bash cluster/setup_cluster_env.sh
# =============================================================================

set -e

echo "=============================================="
echo "FinSight Cluster Environment Setup"
echo "=============================================="

# ── Load modules ──────────────────────────────────────────────────────────────
echo "[1/4] Loading modules..."
module load python/3.11 2>/dev/null || module load python3 2>/dev/null || {
    echo "WARNING: Could not load python module. Using system python."
}
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || {
    echo "WARNING: Could not load cuda module."
}

python3 --version
pip --version

# ── Create virtual environment ────────────────────────────────────────────────
echo ""
echo "[2/4] Creating virtual environment at ~/finsight-llm/venv ..."
mkdir -p ~/finsight-llm
python3 -m venv ~/finsight-llm/venv
source ~/finsight-llm/venv/bin/activate
pip install --upgrade pip --quiet

# ── Install vLLM and dependencies ─────────────────────────────────────────────
echo ""
echo "[3/4] Installing vLLM and huggingface_hub..."
echo "      (This may take 5-10 minutes — vLLM is large)"

# Install vLLM — will auto-select the right CUDA wheels
pip install vllm --quiet

# HuggingFace tools for model download
pip install huggingface_hub hf_transfer --quiet

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "vLLM installed: $(python -c 'import vllm; print(vllm.__version__)')"

# ── Verify GPU is accessible ──────────────────────────────────────────────────
echo ""
echo "[4/4] Checking GPU access..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
else
    echo "WARNING: nvidia-smi not found on login node (normal — GPUs are on compute nodes)"
fi

# ── Create model directory ────────────────────────────────────────────────────
mkdir -p ~/finsight-llm/models
echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set your HF_TOKEN in cluster/serve_vllm.sh"
echo "     Get token at: https://huggingface.co/settings/tokens"
echo ""
echo "  2. Check your cluster's GPU partition name:"
echo "     sinfo -s"
echo "     Then update #SBATCH --partition= in cluster/serve_vllm.sh"
echo ""
echo "  3. Submit the vLLM job:"
echo "     sbatch cluster/serve_vllm.sh"
echo ""
echo "  4. Watch startup logs:"
echo "     tail -f cluster/logs/vllm_<JOBID>.out"
echo "=============================================="
