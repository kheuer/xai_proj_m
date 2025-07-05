#!/bin/bash -l
#SBATCH --job-name=tune
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module add python

conda activate xAI-Proj

tar xvzf "$WORK/camelyon17.tar.gz" -C "$TMPDIR"

#copy .tar to local ssd and unzip is done in script
#copy only checkpoints to $work during running
python3 /home/woody/barz/barz129h/xai_proj_m/tuner.py --model ResNet18 --pretrained True --transformations False --targetdomain 0

cp -r "$TMPDIR/trials" "$WORK"