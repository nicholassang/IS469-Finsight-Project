# FinSight — GPU Cluster Setup (University HPC / SLURM)

This folder contains everything needed to run the LLM backend on a university
GPU cluster and connect FinSight to it from your laptop.

## How it works

```
Your Laptop                        University HPC Cluster
───────────────────                ─────────────────────────────────
FinSight (Streamlit)               SLURM Job → GPU Node
config: backend=openai    ←──────  vLLM server (OpenAI-compatible API)
base_url=localhost:8000   tunnel   port 8000
                                   Model: Qwen2.5-32B or 14B (auto-selected)
```

vLLM exposes an OpenAI-compatible API, so FinSight uses the `openai` backend
pointed at `localhost:8000` — no code changes needed.

---

## Files in this folder

| File | Purpose | Where to run |
|------|---------|-------------|
| `setup_cluster_env.sh` | One-time: installs vLLM + Python env | Login node |
| `serve_vllm.sh` | SLURM job that starts vLLM | Submitted via sbatch |
| `keep_alive.sh` | Auto-resubmits job when it expires | Login node (tmux) |
| `tunnel.sh` | Creates SSH tunnel to GPU node | Your laptop |

---

## Full Setup — Step by Step

### Step 1 — On your laptop: push the project to the cluster

```bash
# Copy your FinSight project to the cluster
scp -r /path/to/finsight your_username@hpc.university.edu:~/finsight

# Or if you've set up GitHub (preferred):
ssh your_username@hpc.university.edu
git clone https://github.com/YOUR-USERNAME/finsight.git ~/finsight
```

### Step 2 — On the cluster login node: one-time environment setup

```bash
ssh your_username@hpc.university.edu
cd ~/finsight

chmod +x cluster/*.sh
bash cluster/setup_cluster_env.sh
```

This installs vLLM into `~/finsight-llm/venv`. Takes about 5-10 minutes.

### Step 3 — Edit the SLURM script for your cluster

Open `cluster/serve_vllm.sh` and set:

```bash
# Line 26 — your HuggingFace token (get from huggingface.co/settings/tokens)
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Line 12 — your cluster's GPU partition name
#SBATCH --partition=gpu_a100        # check with: sinfo -s
```

To find your partition name:
```bash
sinfo -s          # lists all partitions
sinfo -p gpu      # check if 'gpu' exists
```

To find the right gres (GPU resource) name:
```bash
sinfo -o "%P %G" | grep -i a100    # find what your cluster calls A100s
```
Common values: `gpu:a100:1`, `gpu:1`, `gpu:a100_80gb:1`, `gpu:a100_40gb:1`

### Step 4 — Submit the job

```bash
# From ~/finsight on the cluster:
sbatch cluster/serve_vllm.sh

# Check it's queued/running
squeue -u $USER

# Watch the startup logs (replace JOBID with your actual job ID)
tail -f cluster/logs/vllm_<JOBID>.out
```

Wait for this line before moving on:
```
INFO:     Application startup complete.
```
This takes about **3-5 minutes** after the job starts (model loading time).

### Step 5 — On your laptop: start the SSH tunnel

```bash
cd ~/finsight
bash cluster/tunnel.sh your_username hpc.university.edu
```

This auto-finds the node and starts the tunnel. You should see:
```
✓ Tunnel is working!
  Model available: qwen2.5-32b
```

Leave this terminal open for the duration of your session.

### Step 6 — Update FinSight config

In `config/settings.yaml`:
```yaml
generation:
  backend: openai
  model: qwen2.5-32b       # must match what the tunnel.sh found
  temperature: 0.0
  max_tokens: 800
```

In `.env`:
```
OPENAI_API_KEY=dummy-not-used
OPENAI_BASE_URL=http://localhost:8000/v1
```

### Step 7 — Launch FinSight

```bash
# On your laptop (new terminal, keep tunnel terminal open):
cd ~/finsight
streamlit run app/streamlit_app.py
```

---

## Keeping the job alive

University jobs have time limits (typically 8-24 hours). Two options:

**Option A — Manual resubmit** (simpler):
```bash
# Check time remaining
squeue -u $USER

# Before it expires, resubmit
sbatch cluster/serve_vllm.sh

# Update tunnel with new node name if it changed
bash cluster/tunnel.sh your_username hpc.university.edu
```

**Option B — Auto keep-alive** (recommended for extended use):
```bash
# SSH into cluster, start a tmux session
ssh your_username@hpc.university.edu
tmux new -s finsight

# Inside tmux:
cd ~/finsight
bash cluster/keep_alive.sh

# Detach with Ctrl+B then D
# Reattach later with: tmux attach -t finsight
```

---

## Model auto-selection by VRAM

The SLURM script detects GPU memory at job start and picks accordingly:

| GPU | VRAM | Model selected | Quality |
|-----|------|---------------|---------|
| A100 80GB | 80 GB | **Qwen2.5-32B-Instruct** | Excellent — strong financial reasoning |
| A100 40GB | 40 GB | **Qwen2.5-14B-Instruct** | Very good — solid for RAG tasks |
| Other | <38 GB | **Qwen2.5-7B-Instruct** | Good — reliable baseline |

All models are downloaded automatically on first run. Download sizes:
- Qwen2.5-32B: ~65 GB
- Qwen2.5-14B: ~28 GB
- Qwen2.5-7B: ~15 GB

Downloads happen **inside the SLURM job** on first run (takes 20-60 min),
then are cached at `~/finsight-llm/models/` for instant loading on resubmit.

---

## Troubleshooting

**Job queued but not starting:**
```bash
squeue -u $USER       # check reason (Resources, Priority, etc.)
sinfo -p gpu          # check if GPUs are available in the partition
```

**"CUDA out of memory" error in logs:**
```bash
# Reduce gpu-memory-utilization in serve_vllm.sh from 0.90 to 0.85
# Or reduce max-model-len from 8192 to 4096
```

**Tunnel connects but curl returns connection refused:**
```bash
# vLLM may still be loading — check logs
tail -f cluster/logs/vllm_<JOBID>.out
# Wait for "Application startup complete"
```

**Module not found errors on cluster:**
```bash
module avail python    # find correct Python module name
module avail cuda      # find correct CUDA module name
# Update module load lines in setup_cluster_env.sh and serve_vllm.sh
```

**Job immediately fails:**
```bash
cat cluster/logs/vllm_<JOBID>.err    # check error log
# Common causes: wrong partition name, wrong gres name, insufficient mem
```

**Model quality issues (hallucinating numbers):**
- Switch `temperature: 0.0` is already set — good
- Check that retrieved chunks actually contain the answer (use the debug panel in Streamlit)
- If on 40GB: consider requesting the 80GB node if your cluster has them (`--gres=gpu:a100_80gb:1`)
