#!/bin/bash
# =============================================================================
# tunnel.sh  — Run this on YOUR LAPTOP (not the cluster)
# Sets up the SSH tunnel from localhost:8000 → cluster GPU node:8000
#
# Usage:
#   bash cluster/tunnel.sh <username> <cluster-login-node>
#
# Example:
#   bash cluster/tunnel.sh jsmith hpc.university.edu
#
# The script will:
#   1. SSH into the cluster and find the node running your vLLM job
#   2. Set up a port-forward tunnel automatically
#   3. Test the connection
# =============================================================================

USER_NAME="${1:?Usage: bash cluster/tunnel.sh <username> <cluster-login-node>}"
CLUSTER="${2:?Usage: bash cluster/tunnel.sh <username> <cluster-login-node>}"
LOCAL_PORT=8000
REMOTE_PORT=8000

echo "=============================================="
echo "FinSight SSH Tunnel Setup"
echo "User:    $USER_NAME"
echo "Cluster: $CLUSTER"
echo "=============================================="

# Find the node running the vLLM job
echo "Finding vLLM job node on cluster..."
NODE=$(ssh "${USER_NAME}@${CLUSTER}" \
    "squeue -u $USER_NAME --name=finsight-vllm --noheader -o '%N' 2>/dev/null | head -1 | tr -d ' '")

if [ -z "$NODE" ] || [ "$NODE" = "N/A" ]; then
    echo ""
    echo "ERROR: No running finsight-vllm job found."
    echo ""
    echo "Check job status with:"
    echo "  ssh ${USER_NAME}@${CLUSTER} 'squeue -u ${USER_NAME}'"
    echo ""
    echo "If the job is still starting, wait 2-3 minutes and try again."
    exit 1
fi

echo "Found vLLM job running on node: $NODE"
echo ""
echo "Starting tunnel: localhost:${LOCAL_PORT} → ${NODE}:${REMOTE_PORT}"
echo "Press Ctrl+C to stop the tunnel."
echo ""

# Test that the port will be reachable before starting tunnel
echo "Connecting..."
ssh -N \
    -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "${USER_NAME}@${CLUSTER}" &

TUNNEL_PID=$!
sleep 3

# Test the connection
if curl -s --max-time 5 "http://localhost:${LOCAL_PORT}/v1/models" > /dev/null 2>&1; then
    echo "✓ Tunnel is working!"
    echo ""
    MODEL=$(curl -s "http://localhost:${LOCAL_PORT}/v1/models" | python3 -c \
        "import sys,json; data=json.load(sys.stdin); print(data['data'][0]['id'])" 2>/dev/null || echo "unknown")
    echo "  Model available: $MODEL"
    echo ""
    echo "FinSight is ready. In a new terminal:"
    echo "  cd your-finsight-folder"
    echo "  streamlit run app/streamlit_app.py"
    echo ""
    echo "Keeping tunnel alive (Ctrl+C to stop)..."
    wait $TUNNEL_PID
else
    echo ""
    echo "WARNING: Tunnel connected but vLLM may still be loading."
    echo "Check cluster logs:"
    echo "  ssh ${USER_NAME}@${CLUSTER} 'tail -f ~/finsight-llm/logs/vllm_*.out 2>/dev/null || tail -f finsight/cluster/logs/vllm_*.out'"
    echo ""
    echo "Keeping tunnel open — test again in 1-2 minutes:"
    echo "  curl http://localhost:${LOCAL_PORT}/v1/models"
    echo ""
    wait $TUNNEL_PID
fi
