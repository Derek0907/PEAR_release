#!/bin/bash

# ======== Basic logging setup ======== #
LOG_FILE="train_log.txt"
echo "======================" > $LOG_FILE
echo "  Training started on: $(date)" >> $LOG_FILE
echo "======================" >> $LOG_FILE

# ======== Training (single run) ======== #
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
    --epochs 65 \
    --batch-size 12 \
    --eval_period 5 \
    --save_dict \
    --save_args \
    --syncBN \
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
    --yaml_path cfg/all.yaml | tee -a $LOG_FILE

echo "=======================" >> $LOG_FILE
echo "  Training completed at: $(date)" >> $LOG_FILE
echo "=======================" >> $LOG_FILE
