#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=100-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=esmfold
#SBATCH -o logs/esmfold_%j.out
#SBATCH -e logs/esmfold_%j.err

date
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate esmfold

cd /home/student/data/llf/esmfold

export CUDA_VISIBLE_DEVICES=1

esm-fold        -i input/avGFP_candidates_clean.fasta \
                -o qc/avgfp_esmfold \
                --max-tokens-per-batch 1 \
                --chunk-size 128



date
~
