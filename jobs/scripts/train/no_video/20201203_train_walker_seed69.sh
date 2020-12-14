#!/bin/bash                                                                                                 
export LD_LIBRARY_PATH=/global/software/sl-7.x86_64/modules/langs/python/3.6/lib

module load python/3.6
module load pytorch

cd /global/scratch/brianyao/DeepRL_Pretraining/src
python train.py \
        --domain_name walker \
        --task_name walk \
        --action_repeat 4 \
        --mode train \
        --num_shared_layers 8 \
        --num_filters 90 \
        --seed 69 \
        --encoder_lr 1e-4 \
        --actor_lr 1e-4 \
        --critic_lr 1e-4 \
        --replay_buffer_size 100000 \
        --train_steps 80000 \
        --encoder_checkpoint /global/scratch/brianyao/DeepRL_Pretraining/pretraining/ik_checkpoints/walker/model_30_0.09176280273681862_0.17047351044722092.pth \
        --work_dir /global/scratch/brianyao/DeepRL_Pretraining/pretraining/logs/walker/ik_seed69_steps80k/ \
        --save_model
