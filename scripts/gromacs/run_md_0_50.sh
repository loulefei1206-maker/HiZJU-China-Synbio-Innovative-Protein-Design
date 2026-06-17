#!/bin/bash  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=500:00:00
#SBATCH --job-name=WT_S2BUT_50ns
date
export CUDA_VISIBLE_DEVICES=1 
gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -v -deffnm md_0_50 >&run_md_0_50.log
date
