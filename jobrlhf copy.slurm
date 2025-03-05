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
export WANDB_PROJECT_NAME='robust_rlhf'
export WANDB_RUN_NAME='P1_10_1'
export WANDB_MODE=offline
export DIR_TO_RM='/net/scratch/jiaweizhang/MaxMinRLHF/checkpoints/training_reward_model_P1_10_1/peft_last_checkpoint/adapter_model.bin' #TODO: change path
export LOCAL_WORLD_SIZE=1

python training/rlhf.py \
    --dataset_name 'data/alpaca_gpt4_10k.json' \
    --model_name $PATH_TO_TULU_CKPT \
    --reward_model_name $DIR_TO_RM \
    --adafactor False --save_freq 10 --output_max_length 512 --batch_size 16 --gradient_accumulation_steps 1 --batched_gen True --ppo_epochs 8 --learning_rate 1.4e-5 --mini_batch_size 16 \
    --early_stopping True --val_dataset_name 'data/koala_eval_50_.json' --val_every_n_steps 10 --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT_NAME --wandb_run_name $WANDB_RUN_NAME 

