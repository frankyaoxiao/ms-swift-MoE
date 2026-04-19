#!/bin/bash
#SBATCH --job-name=IMPORTANT_v5ta2_c80
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --partition=compute
#SBATCH --time=120:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ============ CONFIGURE THESE ============
RUN_NAME="v5_ta2_clean80"
ROLLOUT_SCRIPT="scripts/grpo/v5_ta2_clean80/node_rollout.sh"
TRAIN_SCRIPT="scripts/grpo/v5_ta2_clean80/node_train_megatron_pp2_2node.sh"
ROLLOUT_PORT=29200
MASTER_PORT=29500
# =========================================

set -euo pipefail
cd /home/fxiao/training_hacking/ms-swift

LOGDIR="/home/fxiao/training_hacking/ms-swift/logs/${RUN_NAME}"
mkdir -p $LOGDIR

# Redirect launcher output to run directory
exec > ${LOGDIR}/launcher.out 2> ${LOGDIR}/launcher.err

# Clean up all child processes on exit (SIGTERM, SIGINT, or normal exit)
# This prevents orphaned vLLM/training processes from holding ports
cleanup() {
    echo "Cleaning up child processes..."
    # Kill all srun steps for this job
    scancel --signal=KILL --steps ${SLURM_JOB_ID} 2>/dev/null || true
    # Kill any remaining children of this script
    kill -- -$$ 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT SIGTERM SIGINT

# Parse Slurm node list
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
ROLLOUT_NODE=${NODES[0]}
TRAIN_NODE_0=${NODES[1]}
TRAIN_NODE_1=${NODES[2]}

# Resolve IPs
ROLLOUT_IP=$(srun --overlap --nodes=1 --ntasks=1 -w $ROLLOUT_NODE hostname -I | awk '{print $1}')
MASTER_IP=$(srun --overlap --nodes=1 --ntasks=1 -w $TRAIN_NODE_0 hostname -I | awk '{print $1}')

echo "========================================"
echo "3-Node GRPO Training"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Run name:  ${RUN_NAME}"
echo "  Rollout:   ${ROLLOUT_NODE} (${ROLLOUT_IP}:${ROLLOUT_PORT})"
echo "  Train 0:   ${TRAIN_NODE_0} (${MASTER_IP}) [rank 0]"
echo "  Train 1:   ${TRAIN_NODE_1} [rank 1]"
echo "  Topology:  TP=4, PP=2, EP=8, ETP=1, DP=2"
echo "  Logs:      ${LOGDIR}/"
echo "========================================"

# Step 1: Launch vLLM rollout server on node 0
srun --overlap --nodes=1 --ntasks=1 -w $ROLLOUT_NODE \
    bash $ROLLOUT_SCRIPT \
    > ${LOGDIR}/rollout.out \
    2> ${LOGDIR}/rollout.err &

# Step 2: Wait for vLLM server to be ready
echo "Waiting for vLLM server at ${ROLLOUT_IP}:${ROLLOUT_PORT}..."
READY=0
for i in $(seq 1 180); do
    if curl -s --connect-timeout 2 "http://${ROLLOUT_IP}:${ROLLOUT_PORT}/health" > /dev/null 2>&1 || \
       curl -s --connect-timeout 2 "http://${ROLLOUT_IP}:${ROLLOUT_PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM server ready after ~$((i * 5))s"
        READY=1
        break
    fi
    sleep 5
done

if [ $READY -eq 0 ]; then
    echo "WARNING: vLLM server not responding after 15 min, launching training anyway..."
fi

# Step 3: Launch training on nodes 1-2
export MASTER_ADDR=${MASTER_IP}
export MASTER_PORT=${MASTER_PORT}

NODE_RANK=0 srun --overlap --nodes=1 --ntasks=1 -w $TRAIN_NODE_0 \
    bash $TRAIN_SCRIPT $ROLLOUT_IP \
    > ${LOGDIR}/train-rank0.out \
    2> ${LOGDIR}/train-rank0.err &

NODE_RANK=1 srun --overlap --nodes=1 --ntasks=1 -w $TRAIN_NODE_1 \
    bash $TRAIN_SCRIPT $ROLLOUT_IP \
    > ${LOGDIR}/train-rank1.out \
    2> ${LOGDIR}/train-rank1.err &

echo "All processes launched. Waiting..."
wait
echo "All processes exited."
