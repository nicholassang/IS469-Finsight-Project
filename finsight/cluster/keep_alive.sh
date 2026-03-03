#!/bin/bash
# =============================================================================
# keep_alive.sh
# Watches your vLLM SLURM job and resubmits automatically if it ends.
# Run on the LOGIN NODE in a tmux/screen session so it persists.
#
# Usage:
#   tmux new -s finsight          # start a persistent session
#   bash cluster/keep_alive.sh    # run inside tmux
#   Ctrl+B then D                 # detach (keeps running)
#   tmux attach -t finsight       # reattach later
# =============================================================================

JOB_NAME="finsight-vllm"
SCRIPT="cluster/serve_vllm.sh"
CHECK_INTERVAL=300   # check every 5 minutes

echo "FinSight keep_alive started at $(date)"
echo "Watching for job: $JOB_NAME"
echo "Resubmit script:  $SCRIPT"
echo "Check interval:   ${CHECK_INTERVAL}s"
echo ""

while true; do
    RUNNING=$(squeue -u "$USER" --name="$JOB_NAME" --noheader | wc -l)

    if [ "$RUNNING" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Job not running — resubmitting..."
        NEW_JOB=$(sbatch "$SCRIPT" | awk '{print $4}')
        echo "[$(date '+%H:%M:%S')] Submitted new job ID: $NEW_JOB"
        echo ""
        echo "  → Update your SSH tunnel once the job starts:"
        echo "    squeue -u \$USER  (get the new node name)"
        echo "    ssh -N -L 8000:<NODENAME>:8000 \$USER@<CLUSTER>"
    else
        NODE=$(squeue -u "$USER" --name="$JOB_NAME" --noheader -o "%N" | head -1)
        TIMELEFT=$(squeue -u "$USER" --name="$JOB_NAME" --noheader -o "%L" | head -1)
        echo "[$(date '+%H:%M:%S')] Job running on $NODE — time left: $TIMELEFT"
    fi

    sleep "$CHECK_INTERVAL"
done
