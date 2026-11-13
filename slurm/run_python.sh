#!/bin/bash

#SBATCH --job-name=python_job
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  sbatch slurm/run_python.sh <python_module> [arguments]"
    exit 1
fi

MODULE=$1
shift

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "======================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Module      : $MODULE"
echo "Arguments   : $@"
echo "Started     : $(date)"
echo "======================================="

cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"

source .venv/bin/activate

: "${EVENTBENCH_DATASET_DIR:?Set EVENTBENCH_DATASET_DIR before submitting}"
export EVENTBENCH_DATASET_DIR

echo "Dataset directory: $EVENTBENCH_DATASET_DIR"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python -m "$MODULE" "$@"

echo
echo "Finished at $(date)"
echo "======================================="