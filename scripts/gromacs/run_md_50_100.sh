#!/bin/bash  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=500:00:00
#SBATCH --job-name=MD
date
export CUDA_VISIBLE_DEVICES=0
gmx mdrun -ntmpi 1 -ntomp 16 -nb gpu -pme gpu -v -deffnm md_0_50 -s md_50_100.tpr -cpi md_0_50.cpt -noappend >&run_md_50_100.log
date

