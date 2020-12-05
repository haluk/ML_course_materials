#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --job-name=cifar
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint=gpu_16gb
#SBATCH --error=job.cifar.%J.err
#SBATCH --output=job.cifar.%J.out

module load anaconda
conda activate hw

python -u $@
