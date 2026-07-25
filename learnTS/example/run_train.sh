#!/bin/bash
# Recorded training and inference commands for LearnTS.
#
# The hyperparameters below are the ones the reported runs used; they were
# previously baked into submit_script.sh. Two GPUs are requested because the
# run used --devices 2 --strategy ddp.

CONDA_ENV="${CONDA_ENV:-learnts}"

# Training (2 GPUs)
GPUS_PER_TASK=2 sbatch --gres=gpu:2 ../run.slurm "$CONDA_ENV" train.py \
    --accelerator gpu --devices 2 --strategy ddp \
    --learning_rate 2e-4 --gru_dropout 0.4 --num_pair_embedding 128 \
    --max_epoch 3000 --gradient_clip_val 1.0 --log_every_n_steps 10 --precision 64

# Inference (1 GPU) -- set the checkpoint inside inference_modified.py first
# sbatch --mem=60G ../run.slurm "$CONDA_ENV" inference_modified.py
