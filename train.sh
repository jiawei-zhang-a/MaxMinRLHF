export OUTPUT_DIR="./checkpoints/tulu_2_PPO_P1_10_1/" #TODO: change path
export PATH_TO_LLAMDA="allenai/tulu-2-7b" #TODO: change path
export WANDB_PROJECT_NAME='robust_rlhf'
export WANDB_RUN_NAME='P1_10_1'
export WANDB_MODE=offline
export DIR_TO_RM='/home/jiaweizhang/MaxMinRLHF/checkpoints/training_reward_model_P1_10_1/peft_last_checkpoint/adapter_model.bin' #TODO: change path
export LOCAL_WORLD_SIZE=1

python training/rlhf.py \
    --dataset_name 'data/alpaca_gpt4_10k.json' \
    --model_name $PATH_TO_LLAMDA \
    --reward_model_name $DIR_TO_RM \
    --adafactor False --save_freq 10 --output_max_length 512 --batch_size 1 --gradient_accumulation_steps 1 --batched_gen True --ppo_epochs 8 --learning_rate 1.4e-5 --mini_batch_size 1 \
    --early_stopping True --val_dataset_name 'data/koala_eval_50_.json' --val_every_n_steps 10 --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT_NAME --wandb_run_name $WANDB_RUN_NAME 