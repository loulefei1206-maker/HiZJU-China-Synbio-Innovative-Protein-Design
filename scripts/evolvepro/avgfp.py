from evolvepro.src.evolve import evolve_experimental, evolve_experimental_multi
import os

### codes modified by Ling Jiang, ling.jiang@zju.edu.cn ###

# Find the current directory
current_directory = os.getcwd()

protein_name = 'avgfp'
embeddings_base_path = output_dir = os.path.join(current_directory, '../output/plm/esm')
embeddings_file_name = 'avgfp_esm2_t36_3B_UR50D.csv'
round_base_path = output_dir = os.path.join(current_directory, '../data/exp/exp_data/rounds')
wt_fasta_path = output_dir = os.path.join(current_directory, "../data/exp/wt_fasta/avgfp_WT.fasta")
number_of_variants = 50
output_dir = output_dir = os.path.join(current_directory, '../output/exp_results/')

# Round 1
round_name = 'ROUND1'
round_file_names = ['avgfp ROUND1.xlsx']
rename_WT = True

evolve_experimental(
    protein_name,
    round_name,
    embeddings_base_path,
    embeddings_file_name,
    round_base_path,
    round_file_names,
    wt_fasta_path,
    rename_WT,
    number_of_variants,
    output_dir
)

