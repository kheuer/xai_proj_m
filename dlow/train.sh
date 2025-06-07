#!/bin/bash

# Load conda into the shell
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate dlow
nohup python train.py \
--dataroot ./datasets \
--name projectname \
--dataset_mode unaligned \
--model cycle_gan \
--checkpoints_dir ./checkpoints \
--batchSize 2 \
--gpu_ids 0 \
--lambda_identity 0 \
--lambda_GA 0 \
--lambda_GB 0 \
--display_id -1 \
--save_epoch_freq 5 \
--targetdomain photo \
--discord \
> output.log 2>&1 &
