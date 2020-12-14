#!/bin/bash
export LD_LIBRARY_PATH=/global/software/sl-7.x86_64/modules/langs/python/3.6/lib

module load python/3.6
module load pytorch

cd /global/scratch/brianyao/DeepRL_Pretraining/src
python train.py \
    --domain_name reacher \
    --task_name easy \
    --action_repeat 4 \
    --mode train \
    --num_shared_layers 8 \
    --num_filters 90 \
    --seed 69 \
    --encoder_lr 1e-4 \
    --actor_lr 1e-4 \
    --critic_lr 1e-4 \
    --replay_buffer_size 100000 \
    --train_steps 70000 \
    --work_dir /global/scratch/brianyao/DeepRL_Pretraining/pretraining/logs/reduced/reacher_reduced/ik_seed69_size10000 \
    --encoder_checkpoint /global/scratch/brianyao/DeepRL_Pretraining/pretraining/ik_checkpoints/reacher_10000/model_50_0.02117622915554691_0.051516604920228325.pth \
    --save_model