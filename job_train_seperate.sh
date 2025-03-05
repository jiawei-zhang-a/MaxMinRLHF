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

