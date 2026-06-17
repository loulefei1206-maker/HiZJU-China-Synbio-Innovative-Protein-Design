#!/bin/bash  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=500:00:00
#SBATCH --job-name=nvt
date
export CUDA_VISIBLE_DEVICES=1 
gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -v -deffnm nvt >&run_nvt.log
date
