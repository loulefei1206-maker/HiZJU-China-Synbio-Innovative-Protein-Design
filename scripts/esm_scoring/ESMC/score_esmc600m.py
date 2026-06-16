import os
import csv
import torch
import torch.nn.functional as F
from pathlib import Path

from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
import esm.pretrained

LOCAL_PTH_FILE = "/media/data1/weight/checkpoints/esmc_600m_2024_12_v0.pth"

# ======= 🪄 终极断网拦截器 =======
def mock_data_root(*args, **kwargs):
    fake_root = "/tmp/fake_esmc_root"
    fake_weight_dir = os.path.join(fake_root, "data", "weights")
    os.makedirs(fake_weight_dir, exist_ok=True)
    fake_file_path = os.path.join(fake_weight_dir, "esmc_600m_2024_12_v0.pth")
    if not os.path.exists(fake_file_path):
        try:
            os.symlink(LOCAL_PTH_FILE, fake_file_path)
        except FileExistsError:
            pass
    return Path(fake_root)

esm.pretrained.data_root = mock_data_root
# =================================

# ================= 1. 批量任务配置区域 =================
# 这里配置剩下 4 个 GFP 的序列和路径。avGFP 已经在跑了，所以这里不包含它。
TARGETS = {
    "avGFP": {
        "wt_seq": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "fasta": "./fasta/avgfp_clean.fasta",
        "output": "./result/esmc600m_avgfp_scores.csv"
    },
    "amacGFP": {
        "wt_seq": "MSKGEELFTGIVPVLIELDGDVHGHKFSVRGEGEGDADYGKLEIKFICTTGKLPVPWPTLVTTLSYGILCFARYPEHMKMNDFFKSAMPEGYIQERTIFFQDDGKYKTRGEVKFEGDTLVNRIELKGMDFKEDGNILGHKLEYNFNSHNVYIMPDKANNGLKVNFKIRHNIEGGGVQLADHYQTNVPLGDGPVLIPINHYLSCQTAISKDRNETRDHMVFLEFFSACGHTHGMDELYK",
        "fasta": "./fasta/amacgfp_clean.fasta",
        "output": "./result/esmc600m_amacgfp_scores.csv"
    },
    "sfGFP": {
        "wt_seq": "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "fasta": "./fasta/sfgfp_clean.fasta",
        "output": "./result/esmc600m_sfgfp_scores.csv"
    },
    "cgreGFP": {
        "wt_seq": "MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR",
        "fasta": "./fasta/cgregfp_clean.fasta",
        "output": "./result/esmc600m_cgregfp_scores.csv"
    },
    "ppluGFP": {
        "wt_seq": "MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA",
        "fasta": "./fasta/pplugfp_clean.fasta",
        "output": "./result/esmc600m_pplugfp_scores.csv"
    }
}
# =======================================================

def read_fasta(file_path):
    records = {}
    with open(file_path, 'r') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header: records[header] = "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header: records[header] = "".join(seq)
    return records

def find_mutations(wt_seq, mut_seq):
    if len(wt_seq) != len(mut_seq):
        raise ValueError("仅支持等长的点突变替换。")
    mutations = []
    for i, (w, m) in enumerate(zip(wt_seq, mut_seq)):
        if w != m:
            mutations.append((i, w, m))
    return mutations

def main():
    print("Loading built-in Tokenizer from Biohub SDK...")
    tokenizer = EsmSequenceTokenizer()
    
    try:
        mask_token_id = tokenizer.mask_token_id
    except AttributeError:
        mask_token_id = tokenizer.encode("<mask>")[1]
    
    print("Initializing architecture and loading local weights (ONLY ONCE) via interceptor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型只在最外层加载一次！
    model = ESMC.from_pretrained("esmc_600m").eval().to(device)
    
    # 开始遍历 4 个任务
    for target_name, config in TARGETS.items():
        wt_seq = config["wt_seq"]
        fasta_file = config["fasta"]
        output_csv = config["output"]
        
        print(f"\n==================================================")
        print(f"🚀 正在处理任务: {target_name}")
        print(f"==================================================")
        
        if not os.path.exists(fasta_file):
            print(f"⚠️ 找不到文件 {fasta_file}，跳过此任务！")
            continue
            
        print(f"Reading FASTA file: {fasta_file}")
        mutant_records = read_fasta(fasta_file)
        results = []
        
        wt_input_ids_list = tokenizer.encode(wt_seq)
        
        for header, mut_seq in mutant_records.items():
            try:
                mutations = find_mutations(wt_seq, mut_seq)
                if not mutations:
                    results.append({"header": header, "mutations": "WT", "total_llr": 0.0})
                    continue
                    
                total_llr = 0.0
                mut_names = []
                
                for pos, wt_aa, mut_aa in mutations:
                    target_idx = pos + 1 
                    
                    masked_input_ids = wt_input_ids_list.copy()
                    masked_input_ids[target_idx] = mask_token_id
                    masked_tensor = torch.tensor(masked_input_ids).unsqueeze(0).to(device)
                    
                    with torch.inference_mode():
                        output = model(sequence_tokens=masked_tensor)
                        logits = output.sequence_logits[0] if hasattr(output, 'sequence_logits') else output.logits[0]
                        log_probs = F.log_softmax(logits, dim=-1)
                    
                    wt_token_id = tokenizer.encode(wt_aa)[1]
                    mut_token_id = tokenizer.encode(mut_aa)[1]
                    
                    llr = log_probs[target_idx, mut_token_id].item() - log_probs[target_idx, wt_token_id].item()
                    total_llr += llr
                    
                    mut_names.append(f"{wt_aa}{pos+1}{mut_aa}")
                
                mut_str = ", ".join(mut_names)
                print(f"[{target_name}] Processed [{header}] -> {mut_str} | Masked-LLR: {total_llr:.4f}")
                results.append({"header": header, "mutations": mut_str, "total_llr": round(total_llr, 4)})
                
            except Exception as e:
                print(f"Error processing {header}: {e}")
                results.append({"header": header, "mutations": "ERROR", "total_llr": "ERROR"})
                
        # 🌟 安全写入：确保结果文件夹存在
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["header", "mutations", "total_llr"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"✅ {target_name} 打分完成！结果已保存至: {output_csv}")

    print("\n🎉 所有批量任务已全部执行完毕！")

if __name__ == "__main__":
    main()