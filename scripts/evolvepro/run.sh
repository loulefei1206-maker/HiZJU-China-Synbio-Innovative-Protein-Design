#!/bin/bash
# Configuration values for SLURM job submission.
# One leading hash ahead of the word SBATCH is not a comment, but two are.
#SBATCH --time=2400:00:00 
#SBATCH --job-name=evolvepro
#SBATCH -N 1   
#SBATCH -n 12
#SBATCH --mem=100gb  

### codes modified by Ling Jiang, ling.jiang@zju.edu.cn ###

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate evolvepro

python avgfp.py > results.dat
