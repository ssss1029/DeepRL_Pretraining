#!/bin/bash
#SBATCH --job-name=debug285
#SBATCH --account=fc_bioml
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_debug
#SBATCH --time=05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/global/scratch/brianyao/DeepRL_Pretraining/logs/TEMP/debug285.out

# Assume we are in DeepRL_Pretraining

export LD_LIBRARY_PATH=/global/software/sl-7.x86_64/modules/langs/python/3.6/lib

module load python/3.6
module load pytorch
cd /global/scratch/brianyao/DeepRL_Pretraining

python src/train.py \
    --domain_name walker \
    --task_name walk \
    --action_repeat 4 \
    --mode train \
    --num_shared_layers 8 \
    --num_filters 90 \
    --seed 2 \
    --encoder_lr 1e-4 \
    --actor_lr 1e-4 \
    --critic_lr 1e-4 \
    --replay_buffer_size 100000 \
    --train_steps 100000 \
    --work_dir logs/TEMP \
    --save_model
