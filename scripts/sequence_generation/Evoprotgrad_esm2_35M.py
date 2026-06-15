import evo_prot_grad
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import os

print("⏳ Step 1: Loading ESM2-35M model expert from local path...")

# 1. Explicitly specify the local folder path
local_model_path = '/home/student/data/llf/esm2_35M_hf'

# 2. Load tokenizer and model with MaskedLM head locally (No internet required)
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(local_model_path, local_files_only=True)

# 3. Wrap it as an expert for evo_prot_grad
esm2_expert = evo_prot_grad.get_expert(
    'esm', 
    'mutant_marginal', 
    temperature = 1.0,
    model = model,
    tokenizer = tokenizer,
    device = 'cuda:1'
)
print("✅ ESM2 local model expert loaded successfully!")

print("\n🚀 Step 3: Starting Directed Evolution simulation...")
print("Notice: Current configuration uses 16 parallel chains, 1000 steps per chain. Please wait...")
variants, scores = evo_prot_grad.DirectedEvolution(
    wt_fasta = 'avgfp.fasta',
    output = 'all',
    experts = [esm2_expert],
    parallel_chains = 16,
    n_steps = 1000,              
    max_mutations = 6,
    verbose = False
)()
print("✅ Directed Evolution simulation completed! Variant sequences and scores obtained.")

print("\n⏳ Step 5: Flattening parallel multi-chain data and removing whitespaces from sequences...")
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Output a standard FASTA file compliant with CD-HIT format
output_filename = os.path.join(data_folder, "esm2_35M_avgfp_evolution.fasta")

flat_variants = []
flat_scores = []

# Parse out sequences and their corresponding scores simultaneously
if len(variants) > 0 and isinstance(variants[0], list):
    for chain_vars, chain_scores in zip(variants, scores):
        for var_seq, score in zip(chain_vars, chain_scores):
            # Core fix: Use .replace(" ", "") to remove all spaces between amino acids
            clean_seq = str(var_seq).replace(" ", "")
            flat_variants.append(clean_seq)
            flat_scores.append(score)
else:
    for var_seq, score in zip(variants, scores):
        clean_seq = str(var_seq).replace(" ", "")
        flat_variants.append(clean_seq)
        flat_scores.append(score)
print(f"✅ Data processing completed. Parsed {len(flat_variants)} clean variant sequences.")

print(f"\n⏳ Step 6: Writing variant sequences to CD-HIT compatible FASTA file...")
# Write standard FASTA format, with no spaces in headers or sequences
with open(output_filename, mode='w', encoding='utf-8') as fasta_file:
    for i, (seq, score) in enumerate(zip(flat_variants, flat_scores)):
        # Header format: >Seq_1_Score_13.0874 (no spaces)
        fasta_file.write(f">Seq_{i+1}_Score_{score:.4f}\n")
        # Write the compact sequence without spaces
        fasta_file.write(f"{seq}\n")

print(f"🎉 [Success] The final CD-HIT compatible FASTA file has been saved to: {output_filename}")
