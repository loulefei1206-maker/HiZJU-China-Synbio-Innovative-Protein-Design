#!/bin/bash  
##### Node number, we only have 1 node called havana
#SBATCH --nodes=1
##### Number of CPUs for each node, this jobs will use 8 CPUs.
#SBATCH --ntasks-per-node=16
##### The time of this job, this job will be terminated after 24 hours.
#SBATCH --time=24000:00:00
##### The job name for display
#SBATCH --job-name=esm_fasta

##### The beginning date and time of this job
date
source activate mlde3
export CUDA_VISIBLE_DEVICES=0

# ==================== 用户配置区 ====================
# 1. 这里填入您的突变体 FASTA 文件的绝对或相对路径
INPUT_FASTA="./您的突变体文件.fasta" 

# 2. 结果输出的 CSV 文件名
OUTPUT_CSV="WT-marginals_fasta.csv"
# ===================================================

python predict.py \
    --model-location esm1v_t33_650M_UR90S_1 esm1v_t33_650M_UR90S_2 esm1v_t33_650M_UR90S_3 esm1v_t33_650M_UR90S_4 esm1v_t33_650M_UR90S_5 esm2_t33_650M_UR50D \
    --sequence MDKKPLNTLISATGLWMSRTGTIHKIKHHEVSRSKIYIEMACGDHLVVNNSRSSRTARALRHHKYRKTCKRCRVSDEDLNKFLTKANEDQTSVKVKVVSAPTRTKKAMPKSVARAPKPLENTEAAQAQPSGSKFSPAIPVSTQESVSVPASVSTSISSISTGATASALVKGNTNPITSMSAPVQASAPALTKSQTDRLEVLLNPKDEISLNSGKPFRELESELLSRRKKDLQQIYAEERENYLGKLEREITRFFVDRGFLEIKSPILIPLEYIERMGIDNDTELSKQIFRVDKNFCLRPMLAPNLYNYLRKLDRALPDPIKIFEIGPCYRKESDGKEHLEEFTMLNFCQMGSGCTRENLESIITDFLNHLGIDFKIVGDSCMVYGDTLDVMHGDLELSSAVVGPIPLDREWGIDKPWIGAGFGLERLLKVKHDFKNIKRAARSESYYNGISTNL \
    --fasta-input $INPUT_FASTA \
    --mutation-col mutant \
    --dms-output $OUTPUT_CSV \
    --offset-idx 1 \
    --scoring-strategy wt-marginals