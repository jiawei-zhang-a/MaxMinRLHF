#!/bin/bash
#
#SBATCH --mail-user=jiaweizhang@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=general        # Use the 'general' partition
#SBATCH --gres=gpu
#SBATCH --mem=32G
#SBATCH --time=10:30:00            # Set a time limit of 3 minutes
#SBATCH --job-name=OM-RLHF  # Name the job
#SBATCH --output=slurm-%j.out       # Output file (stdout)
#SBATCH --error=slurm-%j.err        # Error file (stderr)

# Ensure Miniconda is initialized correctly
eval "$(~/miniconda3/bin/conda shell.bash hook)"  # Adjust path if needed
conda activate rlhf

# Debugging: Print current Conda environment
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

#print GPU
nvidia-smi

# P1A,P1B,P2A,P2B,P3A,P3B
export OUTPUT_DIR="./checkpoints/tulu2_P1A_Epoch1" #TODO: change path
export PATH_TO_TULU_CKPT="/net/scratch/jiaweizhang/" #TODO:can be changed to llama2 7B
export PATH_TO_RM_DATA='/net/scratch/jiaweizhang/MaxMinRLHF/data/rm_training/P1A.json' #TODO: change path
export EVAL_DATASET_NAME='/net/scratch/jiaweizhang/MaxMinRLHF/data/rm_training/P1A.json' #TODO: change path

python training_reward_model.py \
    --model_name $PATH_TO_TULU_CKPT \
    --dataset_name $PATH_TO_RM_DATA \
    --eval_dataset_name $EVAL_DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 

