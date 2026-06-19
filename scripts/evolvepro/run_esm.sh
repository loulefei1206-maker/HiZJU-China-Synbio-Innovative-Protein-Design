#!/bin/bash
#SBATCH --time=2400:00:00 
#SBATCH --job-name=esm
#SBATCH -N 1   
#SBATCH -n 16 
#SBATCH --mem=200gb  

### codes modified by Ling Jiang, ling.jiang@zju.edu.cn ###

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate plm

# Get the current directory
current_directory=$(pwd)

study_names=("avgfp")
model_names=("esm2_t36_3B_UR50D")
fasta_path="${current_directory}/../output/exp/"
results_path="${current_directory}/../output/plm/esm/"

repr_layers=36
toks_per_batch=16

mkdir -p ${results_path}

for model_name in "${model_names[@]}"; do
  for study in "${study_names[@]}"; do
    command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python3 /opt/software/EvolvePro/evolvepro/plm/esm/extract.py ${model_name} ${fasta_path}${study}.fasta ${results_path}${study}/${model_name} --toks_per_batch ${toks_per_batch} --include mean --concatenate_dir ${results_path}"
    echo "Running command: ${command}"
    #echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python3 /opt/software/EvolvePro/evolvepro/plm/esm/extract.py ${model_name} ${fasta_path}${study}.fasta ${results_path}${study}/${model_name} --toks_per_batch ${toks_per_batch} --include mean --concatenate_dir ${results_path}"
    eval "${command}"
  done
done
