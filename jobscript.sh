#!/bin/bash -l
#SBATCH --job-name=tune
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module add python

conda activate xAI-Proj

tar xf "$WORK/camelyon17.tar.gz" -C "$TMPDIR"
cp "$WORK/xai_proj_m/resnet18.pth" "$TMPDIR"
#copy .tar to local ssd and unzip is done in script
#copy only checkpoints to $work during running
#python3 /home/woody/barz/barz129h/xai_proj_m/tuner.py --model ResNet18 --pretrained True --transformations False --targetdomain 0
/home/woody/barz/barz129h/xai_proj_m/jobs.sh

cp -r "$TMPDIR/trials" "$WORK"