# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

class Config:
    """
    Global class that houses all configurations
    """
    
    # Shared args to put onto all of the JOBS
    SHARED_ARGS = ""

    HEADER = "conda activate pad; "

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        
        #####################################################################################
        #### No Pretraining, PixelEncoderFullGroupConvBigger, clean environment
        #####################################################################################
        # "walker_walk_noss_no_pretraining_PixelEncoderFullGroupConvBigger_seed0" : "python3 src/train.py \
        #         --domain_name walker \
        #         --task_name walk \
        #         --action_repeat 4 \
        #         --mode train \
        #         --num_shared_layers 8 \
        #         --num_filters 288 \
        #         --seed 2 \
        #         --encoder_lr 1e-4 \
        #         --actor_lr 1e-4 \
        #         --critic_lr 1e-4 \
        #         --replay_buffer_size 100000 \
        #         --train_steps 100000 \
        #         --work_dir logs/walker_walk/no_ss/no_pretraining_PixelEncoderFullGroupConvBigger_seed0 \
        #         --save_model",
        
        # "cheetah_run_noss_no_pretraining_PixelEncoderFullGroupConvBigger_seed0" : "python3 src/train.py \
        #         --domain_name cheetah \
        #         --task_name run \
        #         --action_repeat 4 \
        #         --mode train \
        #         --num_shared_layers 8 \
        #         --num_filters 288 \
        #         --seed 2 \
        #         --encoder_lr 1e-4 \
        #         --actor_lr 1e-4 \
        #         --critic_lr 1e-4 \
        #         --replay_buffer_size 100000 \
        #         --train_steps 100000 \
        #         --work_dir logs/cheetah_run/no_ss/no_pretraining_PixelEncoderFullGroupConvBigger_seed0 \
        #         --save_model",
        
        # "reacher_easy_noss_no_pretraining_PixelEncoderFullGroupConvBigger_seed0" : "python3 src/train.py \
        #         --domain_name reacher \
        #         --task_name easy \
        #         --action_repeat 4 \
        #         --mode train \
        #         --num_shared_layers 8 \
        #         --num_filters 288 \
        #         --seed 2 \
        #         --encoder_lr 1e-4 \
        #         --actor_lr 1e-4 \
        #         --critic_lr 1e-4 \
        #         --replay_buffer_size 100000 \
        #         --train_steps 100000 \
        #         --work_dir logs/reacher_easy/no_ss/no_pretraining_PixelEncoderFullGroupConvBigger_seed0 \
        #         --save_model",

        # #####################################################################################
        # #### No Pretraining, PixelEncoderFullGroupConvBigger, video_hard environment
        # #####################################################################################
        # "walker_walk_noss_no_pretraining_PixelEncoderFullGroupConvBigger_video_hard__video0_seed2" : "python3 src/train.py \
        #         --domain_name walker \
        #         --task_name walk \
        #         --action_repeat 4 \
        #         --mode video_hard__video0 \
        #         --num_shared_layers 8 \
        #         --num_filters 288 \
        #         --seed 2 \
        #         --encoder_lr 1e-4 \
        #         --actor_lr 1e-4 \
        #         --critic_lr 1e-4 \
        #         --replay_buffer_size 100000 \
        #         --train_steps 100000 \
        #         --work_dir logs/walker_walk/video_hard__video0/no_pretraining_PixelEncoderFullGroupConvBigger_seed2 \
        #         --save_model",
        
        # "cheetah_run_noss_no_pretraining_PixelEncoderFullGroupConvBigger_video_hard__video1_seed2" : "python3 src/train.py \
        #         --domain_name cheetah \
        #         --task_name run \
        #         --action_repeat 4 \
        #         --mode video_hard__video1 \
        #         --num_shared_layers 8 \
        #         --num_filters 288 \
        #         --seed 2 \
        #         --encoder_lr 1e-4 \
        #         --actor_lr 1e-4 \
        #         --critic_lr 1e-4 \
        #         --replay_buffer_size 100000 \
        #         --train_steps 100000 \
        #         --work_dir logs/cheetah_run/video_hard__video1/no_pretraining_PixelEncoderFullGroupConvBigger_seed2 \
        #         --save_model",
        
        # "reacher_easy_noss_no_pretraining_PixelEncoderFullGroupConvBigger_video_hard__video99_seed2" : "python3 src/train.py \
        #         --domain_name reacher \
        #         --task_name easy \
        #         --action_repeat 4 \
        #         --mode video_hard__video99 \
        #         --num_shared_layers 8 \
        #         --num_filters 288 \
        #         --seed 2 \
        #         --encoder_lr 1e-4 \
        #         --actor_lr 1e-4 \
        #         --critic_lr 1e-4 \
        #         --replay_buffer_size 100000 \
        #         --train_steps 100000 \
        #         --work_dir logs/reacher_easy/video_hard__video99/no_pretraining_PixelEncoderFullGroupConvBigger_seed2 \
        #         --save_model",

        ###########################################################################################################
        #### RERUN after weight init bug: 3 Pretrained agents with PixelEncoderFullGroupConvBigger on clean env
        ###########################################################################################################
        "walker_walk_imagenetSupervised_pretrain_PixelEncoderFullGroupConvBigger_seed2__RETRY_weightInitBug" : "python3 src/train.py \
                --domain_name walker \
                --task_name walk \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --num_filters 288 \
                --seed 2 \
                --encoder_lr 1e-4 \
                --actor_lr 1e-4 \
                --critic_lr 1e-4 \
                --replay_buffer_size 100000 \
                --train_steps 100000 \
                --work_dir logs/walker_walk/no_ss/imagenetSupervised_pretrain_PixelEncoderFullGroupConvBigger_seed2__RETRY_weightInitBug \
                --encoder_checkpoint /home/saurav/DeepRL_Pretraining/pretraining/checkpoints/imagenet_supervised__PixelEncoderFullGroupConvBigger/checkpoint_epoch412.pth.tar \
                --save_model",
        
        "cheetah_run_imagenetSupervised_pretrain_PixelEncoderFullGroupConvBigger_seed2__RETRY_weightInitBug" : "python3 src/train.py \
                --domain_name cheetah \
                --task_name run \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --num_filters 288 \
                --seed 2 \
                --encoder_lr 1e-4 \
                --actor_lr 1e-4 \
                --critic_lr 1e-4 \
                --replay_buffer_size 100000 \
                --train_steps 100000 \
                --work_dir logs/cheetah_run/no_ss/imagenetSupervised_pretrain_PixelEncoderFullGroupConvBigger_seed2__RETRY_weightInitBug \
                --encoder_checkpoint /home/saurav/DeepRL_Pretraining/pretraining/checkpoints/imagenet_supervised__PixelEncoderFullGroupConvBigger/checkpoint_epoch412.pth.tar \
                --save_model",
        
        "reacher_easy_imagenetSupervised_pretrain_PixelEncoderFullGroupConvBigger_seed2__RETRY_weightInitBug" : "python3 src/train.py \
                --domain_name reacher \
                --task_name easy \
                --action_repeat 4 \
                --mode train \
                --num_shared_layers 8 \
                --num_filters 288 \
                --seed 2 \
                --encoder_lr 1e-4 \
                --actor_lr 1e-4 \
                --critic_lr 1e-4 \
                --replay_buffer_size 100000 \
                --train_steps 100000 \
                --work_dir logs/reacher_easy/no_ss/imagenetSupervised_pretrain_PixelEncoderFullGroupConvBigger_seed2__RETRY_weightInitBug \
                --encoder_checkpoint /home/saurav/DeepRL_Pretraining/pretraining/checkpoints/imagenet_supervised__PixelEncoderFullGroupConvBigger/checkpoint_epoch412.pth.tar \
                --save_model",

    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 15

    # Minimum memory required on a GPU to consider putting a job on it (MiB).
    MIN_MEMORY_REQUIRED = 4000


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value

def select_gpu(GPUs):
    """
    Select the next best available GPU to run on. If nothing exists, return None
    """
    GPUs = list(filter(lambda gpu: gpu.memoryFree > Config.MIN_MEMORY_REQUIRED, GPUs))
    if len(GPUs) == 0:
        return None
    GPUs = sorted(GPUs, key=lambda gpu: gpu.memoryFree)
    return GPUs[-1]

for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Finding GPU for command \"{0}\"".format(command))
    curr_gpu = select_gpu(GPUtil.getGPUs())

    if curr_gpu == None:
        print("No available GPUs found. Exiting.")
        sys.exit(1)

    print("SUCCESS! Found GPU id = {0} which has {1} MiB free memory".format(curr_gpu.id, curr_gpu.memoryFree))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys '{2} CUDA_VISIBLE_DEVICES={0} {1}' C-m".format(
        curr_gpu.id, command, Config.HEADER
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)
