#!/bin/bash

# SLURM Directives
#####################################
#SBATCH -N 1                            # Number of Nodes
#SBATCH --ntasks-per-node=1             # Number of cores per node.
#SBATCH --partition=dgx                 # Partition (GPU in this case).
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=60G
#SBATCH -J OARD_test                    # Job Name. CHANGE as needed.
#SBATCH -t 120:00:00                     # Time limit for the job.

# Usage Information
#####################################
usage() {
    echo
    echo "Usage: sbatch $0 CONDA_ENV PYTHON_SCRIPT [PYTHON_ARGS...]"
    echo "Example: sbatch $0 my_env my_script.py arg1 arg2"
    echo
    exit 1
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
fi

# Check arguments
#####################################
if [ "$#" -lt 2 ]; then
    echo "ERROR: Insufficient arguments provided."
    usage
fi

# Assign arguments
CONDA_ENV="$1"
PYTHON_SCRIPT="$2"
shift 2  # Remaining arguments will be passed to the Python script



# Change to the directory where the job was submitted
#####################################
cd "$SLURM_SUBMIT_DIR"



# Job Information Logging
#####################################
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs allocated: $(nvidia-smi --list-gpus | wc -l)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "GPU Memory Status"
nvidia-smi
echo "Conda Environment: $CONDA_ENV"
echo "Python Script: $PYTHON_SCRIPT"
echo "========================================="
echo

# Initialize conda
eval "$(conda shell.bash hook)" || { echo "ERROR: Conda initialization failed"; exit 1; }

# Activate the provided Conda environment
echo "Activating Conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Check if the Python Script Exists
#####################################
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script '$PYTHON_SCRIPT' not found!"
    conda deactivate
    exit 1
fi

# Run the Python Script with any additional arguments
#####################################
echo "Running Python script..."
LOG_FILE="${PYTHON_SCRIPT%.*}_${SLURM_JOB_ID}_output.log"
echo "Logging output to: $LOG_FILE"
srun python "$PYTHON_SCRIPT" "$@" > "$LOG_FILE" 2>&1

# Cleanup: Deactivate the Conda Environment
#####################################
echo "Deactivating Conda environment..."
conda deactivate

# Final Logging
#####################################
echo
echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="
