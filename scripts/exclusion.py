import pandas as pd

def filter_fasta_pure_python(input_fasta, exclusion_csv, output_fasta):
    print("Loading exclusion list...")
    df = pd.read_csv(exclusion_csv)
    
    exclude_seqs = set(df['Sequence'].astype(str).str.strip())
    print("Loaded " + str(len(exclude_seqs)) + " sequences to exclude.")

    count_total = 0
    count_kept = 0
    
    print("Filtering fasta file...")
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        current_id = None
        current_seq_lines = []
        
        for line in infile:
            if line.startswith('>'):
                if current_id is not None:
                    count_total += 1
                    full_seq = "".join(current_seq_lines).replace('\n', '').replace('\r', '').strip()
                    
                    if full_seq not in exclude_seqs:
                        outfile.write(current_id + "\n" + "".join(current_seq_lines))
                        count_kept += 1
                
                current_id = line.strip()
                current_seq_lines = []
            else:
                current_seq_lines.append(line)
        
        if current_id is not None:
            count_total += 1
            full_seq = "".join(current_seq_lines).replace('\n', '').replace('\r', '').strip()
            if full_seq not in exclude_seqs:
                outfile.write(current_id + "\n" + "".join(current_seq_lines))
                count_kept += 1

    print("\n--- Filter Finished ---")
    print("Total sequences read: " + str(count_total))
    print("Remaining sequences kept: " + str(count_kept))
    print("Result saved to: " + str(output_fasta))

filter_fasta_pure_python(
    input_fasta="cd-hit_amacgfp",               
    exclusion_csv="Exclusion_List.csv",    
    output_fasta="amacgfp_clean.fasta"    
)
