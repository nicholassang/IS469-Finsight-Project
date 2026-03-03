#!/bin/bash
# =============================================================================
# FinSight — vLLM Server on University HPC (Single A100)
# =============================================================================
# Automatically detects GPU VRAM and selects the best model that fits.
#
# BEFORE SUBMITTING:
#   1. Set YOUR_USERNAME and HF_TOKEN below
#   2. Check your cluster's partition name: sinfo -s
#   3. Check your CUDA module name: module avail cuda
#   4. Submit with: sbatch cluster/serve_vllm.sh
#
# AFTER SUBMITTING:
#   - Watch logs:  tail -f cluster/logs/vllm_<JOBID>.out
#   - Get node:    squeue -u $USER
#   - SSH tunnel:  ssh -N -L 8000:<NODENAME>:8000 <USER>@<CLUSTER>
# =============================================================================

#SBATCH --job-name=finsight-vllm
#SBATCH --partition=gpu              # ← CHANGE to your GPU partition (check: sinfo -s)
#SBATCH --gres=gpu:a100:1            # ← 1x A100 (try gpu:1 if this fails)
#SBATCH --mem=80G                    # safe for both 40GB and 80GB A100
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00              # 8 hours — resubmit if needed
#SBATCH --output=cluster/logs/vllm_%j.out
#SBATCH --error=cluster/logs/vllm_%j.err

# =============================================================================
# USER CONFIG — edit these two lines
# =============================================================================
HF_TOKEN="hf_YOUR_TOKEN_HERE"        # from huggingface.co/settings/tokens
MODEL_DIR="$HOME/finsight-llm/models"
# =============================================================================

set -e
mkdir -p cluster/logs "$MODEL_DIR"

# ── Load cluster modules ──────────────────────────────────────────────────────
# Adjust module names to match your cluster (check: module avail)
module load python/3.11    2>/dev/null || module load python3 2>/dev/null || true
module load cuda/12.1      2>/dev/null || module load cuda   2>/dev/null || true

source ~/finsight-llm/venv/bin/activate

echo "=============================================="
echo "FinSight vLLM Server"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# ── Detect GPU VRAM ───────────────────────────────────────────────────────────
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
VRAM_GB=$((VRAM_MB / 1024))
echo "Detected VRAM: ${VRAM_GB} GB (${VRAM_MB} MB)"

# ── Select model based on VRAM ────────────────────────────────────────────────
#
#  A100 80GB  → Qwen2.5-32B in full bfloat16  (~65GB VRAM)
#               Best quality, fits comfortably, no quantization needed
#
#  A100 40GB  → Qwen2.5-14B in full bfloat16  (~28GB VRAM)
#               Very strong quality, fits with headroom for KV cache
#               Alternative: Llama-3.1-8B (faster, less accurate)
#
if [ "$VRAM_GB" -ge 75 ]; then
    echo "→ A100 80GB detected — using Qwen2.5-32B-Instruct (full precision)"
    MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
    MODEL_PATH="$MODEL_DIR/qwen2.5-32b"
    SERVED_NAME="qwen2.5-32b"
    DTYPE="bfloat16"
    QUANTIZATION=""
    MAX_MODEL_LEN=8192
    GPU_MEM_UTIL=0.90

elif [ "$VRAM_GB" -ge 38 ]; then
    echo "→ A100 40GB detected — using Qwen2.5-14B-Instruct (full precision)"
    MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
    MODEL_PATH="$MODEL_DIR/qwen2.5-14b"
    SERVED_NAME="qwen2.5-14b"
    DTYPE="bfloat16"
    QUANTIZATION=""
    MAX_MODEL_LEN=8192
    GPU_MEM_UTIL=0.88

else
    echo "→ Smaller GPU detected (${VRAM_GB}GB) — using Qwen2.5-7B-Instruct"
    MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    MODEL_PATH="$MODEL_DIR/qwen2.5-7b"
    SERVED_NAME="qwen2.5-7b"
    DTYPE="bfloat16"
    QUANTIZATION=""
    MAX_MODEL_LEN=8192
    GPU_MEM_UTIL=0.85
fi

echo "Model:   $MODEL_NAME"
echo "Path:    $MODEL_PATH"
echo "=============================================="

# ── Download model if not already present ────────────────────────────────────
if [ ! -d "$MODEL_PATH" ] || [ -z "$(ls -A $MODEL_PATH 2>/dev/null)" ]; then
    echo "Model not found locally — downloading from HuggingFace..."
    echo "This may take 20-60 minutes depending on model size and network speed."

    export HF_TOKEN="$HF_TOKEN"
    huggingface-cli download \
        "$MODEL_NAME" \
        --local-dir "$MODEL_PATH" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"

    echo "Download complete."
else
    echo "Model already downloaded at $MODEL_PATH — skipping download."
fi

# ── Launch vLLM server ────────────────────────────────────────────────────────
echo ""
echo "Starting vLLM server..."
echo "Once you see 'Application startup complete', run this on your laptop:"
echo ""
echo "  ssh -N -L 8000:$(hostname):8000 \$USER@<cluster-login-node>"
echo ""

VLLM_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --host 0.0.0.0
    --port 8000
    --trust-remote-code
    --disable-log-requests         # cleaner logs during RAG usage
)

# Add quantization flag if set
if [ -n "$QUANTIZATION" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
