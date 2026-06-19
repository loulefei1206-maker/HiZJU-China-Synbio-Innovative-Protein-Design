import os
from evolvepro.src.process import generate_wt, generate_single_aa_mutants, generate_n_mutant_combinations

### codes modified by Ling Jiang, ling.jiang@zju.edu.cn ###

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

# Define the WT sequences for all proteins
wt_sequences = {
    'avgfp': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
}

# Generate single amino acid mutants for FgDAAO
print('Generating single amino acid mutants for avgfp...')
avgfp_wt_fasta = os.path.join(project_root, 'data', 'exp', 'wt_fasta', 'avgfp_WT.fasta')
generate_wt(wt_sequences['avgfp'], avgfp_wt_fasta)
avgfp_output_file = os.path.join(project_root, 'output', 'exp', 'avgfp.fasta')
generate_single_aa_mutants(avgfp_wt_fasta, avgfp_output_file)

