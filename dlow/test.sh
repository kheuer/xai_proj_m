#!/bin/bash

# Load conda into the shell
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate dlow

python test.py \
--dataroot ./datasets/ \
--name projectname \
--model test \
--dataset_mode single \
--phase test \
--gpu_id 0 \
--checkpoints_dir ./checkpoints/ckpt_cartoon \
--loadSize 256 \
--which_epoch latest \
--which_direction AtoB \
--how_many 20