#!/bin/bash -l
#SBATCH --job-name=tune
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=4:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module add python

conda activate xAI-Proj

tar xf "$WORK/camelyon17.tar.gz" -C "$TMPDIR"
#cp "$WORK/xai_proj_m/resnet18.pth" "$TMPDIR"
#cp -r "$WORK/studies" "$TMPDIR"
#copy .tar to local ssd and unzip is done in script
#copy only checkpoints to $work during running
/home/woody/barz/barz129h/xai_proj_m/jobs.sh

cp "$TMPDIR/results_camelyon.csv" "$WORK"
cp "$TMPDIR/results_camelyon_unbalanced.csv" "$WORK"