import re
from pathlib import Path

input_fasta = "input/sfgfp_clean.fasta"
output_fasta = "input/sfGFP_candidates_clean.fasta"
log_file = "qc/fasta_cleaning_log.csv"

valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

records = []
bad_records = []

def read_fasta(path):
    name = None
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seqs)
                name = line[1:].strip()
                seqs = []
            else:
                seqs.append(line.strip())
        if name is not None:
            yield name, "".join(seqs)

for idx, (name, seq) in enumerate(read_fasta(input_fasta)):
    clean_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    seq = seq.upper().replace("-", "").replace(" ", "")
    
    illegal = sorted(set(seq) - valid_aas)
    length = len(seq)

    if illegal or length < 200 or length > 260:
        bad_records.append((name, length, "".join(illegal), "FAIL"))
    else:
        records.append((clean_name, seq))
        bad_records.append((name, length, "".join(illegal), "PASS"))

with open(output_fasta, "w") as f:
    for name, seq in records:
        f.write(f">{name}\n{seq}\n")

with open(log_file, "w") as f:
    f.write("name,length,illegal_AA,status\n")
    for row in bad_records:
        f.write(",".join(map(str, row)) + "\n")

print(f"输入序列数: {len(bad_records)}")
print(f"通过清洗序列数: {len(records)}")
print(f"输出文件: {output_fasta}")
print(f"日志文件: {log_file}")

