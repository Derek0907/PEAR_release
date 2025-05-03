#!/bin/bash

WEIGHTS_PATH="PATH_TO/pear_weights.pth"

LOG_FILE="eval_log.txt"

echo "Starting evaluation at $(date)" | tee -a $LOG_FILE

python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
    --epochs 10 \
    --batch-size 12 \
    --eval_period 1 \
    --cfg cfg/BLIP.yaml \
    --device cuda \
    --world-size 4 \
    --dist-url env:// \
    --optimizer_type adamw \
    --momentum 0.9 \
    --weight_decay 0 \
    --lr_decay_type cos \
    --Init_lr 0.001 \
    --Min_lr 0.0001 \
    --gpu 0 \
    --seed 3407 \
    --int_loss_type hybrid \
    --train_type hybrid \
    --text_encoder blip \
    --save_root PATH_TO_SAVE_ROOT \
    --yaml_path cfg/all.yaml \
    --pretrain \
    --evaluate_only \
    --weights $WEIGHTS_PATH | tee -a $LOG_FILE

echo "Evaluation finished at $(date)" | tee -a $LOG_FILE
