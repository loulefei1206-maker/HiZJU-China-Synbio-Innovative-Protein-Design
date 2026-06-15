#!/bin/bash  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=500:00:00
#SBATCH --job-name=ColabFold
date
colabfold_batch --num-model 5 ./seqs/GFP.fa output/GFP
date
