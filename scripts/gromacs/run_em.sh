#!/bin/bash  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=500:00:00
#SBATCH --job-name=ener_min
date
export CUDA_VISIBLE_DEVICES=0 
gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme cpu -v -deffnm em >&run_em.log
date
