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

export PATH_TO_TULU_CKPT="/net/scratch/jiaweizhang/" #TODO:can be changed to llama2 7B
export datapath1="/net/scratch/jiaweizhang/MaxMinRLHF/data/rm_training/P1A.json" #TODO: change path
export datapath2="/net/scratch/jiaweizhang/MaxMinRLHF/data/rm_training/P1B.json" #TODO: change path
export OUTPUT_DIR="./checkpoints/training_reward_model_P1_10_1/" #TODO: change path
python training_reward_model_single.py --model_name $PATH_TO_TULU_CKPT --dataset_name1 $datapath1 --dataset_name2 $datapath2 --output_dir $OUTPUT_DIR --num_train_epochs 1 --train_subset2 10
#TODO: change ratio by changing train_subset2
