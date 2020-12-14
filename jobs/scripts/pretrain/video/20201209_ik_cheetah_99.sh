#!/bin/bash
export LD_LIBRARY_PATH=/global/software/sl-7.x86_64/modules/langs/python/3.6/lib

module load python/3.6
module load pytorch
cd /global/scratch/brianyao/DeepRL_Pretraining/pretraining

python env_trainer.py cheetah 1 99
