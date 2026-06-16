import os
import csv
import json
import torch
import torch.nn.functional as F
from pathlib import Path

from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization import get_esmc_model_tokenizers
import esm.pretrained

# ==================== 配置区域 ====================
LOCAL_MODEL_DIR = Path("/media/data1/weight/checkpoints/esmc-6b")
MODEL_REGISTRY_NAME = "esmc_6b_2024_12_v0"

TARGETS = {
    "avGFP": {
        "wt_seq": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "fasta": "./fasta/avgfp_clean.fasta",
        "output": "./result/esmc6b_avgfp_scores.csv"
    },
    "amacGFP": {
        "wt_seq": "MSKGEELFTGIVPVLIELDGDVHGHKFSVRGEGEGDADYGKLEIKFICTTGKLPVPWPTLVTTLSYGILCFARYPEHMKMNDFFKSAMPEGYIQERTIFFQDDGKYKTRGEVKFEGDTLVNRIELKGMDFKEDGNILGHKLEYNFNSHNVYIMPDKANNGLKVNFKIRHNIEGGGVQLADHYQTNVPLGDGPVLIPINHYLSCQTAISKDRNETRDHMVFLEFFSACGHTHGMDELYK",
        "fasta": "./fasta/amacgfp_clean.fasta",
        "output": "./result/esmc6b_amacgfp_scores.csv"
    },
    "sfGFP": {
        "wt_seq": "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "fasta": "./fasta/sfgfp_clean.fasta",
        "output": "./result/esmc6b_sfgfp_scores.csv"
    },
    "cgreGFP": {
        "wt_seq": "MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR",
        "fasta": "./fasta/cgregfp_clean.fasta",
        "output": "./result/esmc6b_cgregfp_scores.csv"
    },
    "ppluGFP": {
        "wt_seq": "MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA",
        "fasta": "./fasta/pplugfp_clean.fasta",
        "output": "./result/esmc6b_pplugfp_scores.csv"
    }
}
# ==================================================


def build_esmc_6b(device: torch.device | str = "cpu") -> ESMC:
    """
    从本地 HuggingFace 分片 safetensors 加载 ESMC-6B，
    并注册进 ESM SDK 的 LOCAL_MODEL_REGISTRY。
    """
    try:
        from safetensors.torch import load_file as st_load
    except ImportError:
        raise ImportError("请先安装 safetensors：pip install safetensors")

    # 1. 按 config.json 中的参数构造模型骨架
    print(f"  Building ESMC architecture: d_model=2560, n_heads=40, n_layers=80")
    with torch.device(device):
        model = ESMC(
            d_model=2560,
            n_heads=40,
            n_layers=80,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=True,   # 如果 CUDA 不支持可改为 False
        ).eval()

    # 2. 读取分片索引，收集所有 shard 文件
    index_path = LOCAL_MODEL_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    print(f"  Loading {len(shard_files)} safetensors shards from {LOCAL_MODEL_DIR} ...")

    # 3. 逐分片加载并合并（避免一次性爆显存，用 cpu 中转）
    merged_state_dict = {}
    for i, shard in enumerate(shard_files, 1):
        print(f"    [{i}/{len(shard_files)}] {shard}")
        shard_dict = st_load(str(LOCAL_MODEL_DIR / shard), device="cpu")
        merged_state_dict.update(shard_dict)

    # 4. Key 重映射，两步处理：
    #    step-1  strip "esmc." 前缀
    #            "esmc.embed.weight"  →  "embed.weight"
    #    step-2  HuggingFace 把输出头叫 "lm_head"，SDK 叫 "sequence_head"，做替换
    #            "lm_head.0.weight"  →  "sequence_head.0.weight"
    remapped = {}
    for k, v in merged_state_dict.items():
        new_k = k[len("esmc."):] if k.startswith("esmc.") else k
        new_k = new_k.replace("lm_head.", "sequence_head.", 1)
        remapped[new_k] = v

    # 5. 载入权重，strict=False 便于诊断 key 不匹配
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys  ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        print("  ✅ State dict loaded perfectly, no missing or unexpected keys.")

    # 6. 移到目标设备（已在 bfloat16 下推断，这里不转精度，交给调用方）
    model = model.to(device)
    return model


# 注册自定义 builder 到 ESM SDK 的本地注册表
esm.pretrained.register_local_model(
    MODEL_REGISTRY_NAME,
    build_esmc_6b
)


# ==================== 工具函数 ====================

def read_fasta(file_path):
    records = {}
    with open(file_path, 'r') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records[header] = "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            records[header] = "".join(seq)
    return records


def find_mutations(wt_seq, mut_seq):
    if len(wt_seq) != len(mut_seq):
        raise ValueError("仅支持等长突变。")
    return [(i, w, m) for i, (w, m) in enumerate(zip(wt_seq, mut_seq)) if w != m]


# ==================== 主流程 ====================

def main():
    print("Loading EsmSequenceTokenizer...")
    tokenizer = EsmSequenceTokenizer()

    # 官方推荐：用 get_vocab() 字典查询 token id
    vocab = tokenizer.get_vocab()

    # 获取 mask token id
    try:
        mask_token_id = tokenizer.mask_token_id
        assert mask_token_id is not None
    except (AttributeError, AssertionError):
        mask_token_id = tokenizer.encode("<mask>")[1]
    print(f"mask_token_id = {mask_token_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading {MODEL_REGISTRY_NAME} from local safetensors ...")
    print("⏳ 6B 模型较大，加载可能需要几分钟...")

    # from_pretrained 会调用我们注册的 build_esmc_6b
    model = (
        ESMC.from_pretrained(MODEL_REGISTRY_NAME)
        .eval()
        .to(device, dtype=torch.bfloat16)
    )
    print("✅ 模型加载完成！\n")

    for target_name, config in TARGETS.items():
        wt_seq    = config["wt_seq"]
        fasta_file = config["fasta"]
        output_csv = config["output"]

        print(f"{'='*50}")
        print(f"🚀 [ESMC-6B] 正在处理: {target_name}")
        print(f"{'='*50}")

        if not os.path.exists(fasta_file):
            print(f"⚠️  找不到文件 {fasta_file}，跳过！")
            continue

        print(f"Reading FASTA: {fasta_file}")
        mutant_records = read_fasta(fasta_file)
        results = []

        # 编码 wt 序列（含 BOS/EOS），统一为 list
        wt_input_ids = tokenizer.encode(wt_seq)
        if hasattr(wt_input_ids, 'tolist'):
            wt_input_ids = wt_input_ids.tolist()
        else:
            wt_input_ids = list(wt_input_ids)

        for header, mut_seq in mutant_records.items():
            try:
                mutations = find_mutations(wt_seq, mut_seq)

                if not mutations:
                    results.append({"header": header, "mutations": "WT", "total_llr": 0.0})
                    continue

                total_llr = 0.0
                mut_names = []

                for pos, wt_aa, mut_aa in mutations:
                    # target_idx = pos + 1：跳过 BOS token（官方教程明确说明）
                    target_idx = pos + 1

                    masked_ids = wt_input_ids.copy()
                    masked_ids[target_idx] = mask_token_id
                    masked_tensor = torch.tensor(masked_ids).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        output = model(sequence_tokens=masked_tensor)
                        logits = (
                            output.sequence_logits[0]
                            if hasattr(output, 'sequence_logits')
                            else output.logits[0]
                        )
                        # 转回 fp32 再做 log_softmax，保证数值精度
                        log_probs = F.log_softmax(logits.float(), dim=-1)

                    # 官方推荐：用 vocab 字典取 token id
                    wt_token_id  = vocab[wt_aa]
                    mut_token_id = vocab[mut_aa]

                    # LLR = log P(mut|masked_context) - log P(wt|masked_context)
                    llr = (
                        log_probs[target_idx, mut_token_id].item()
                        - log_probs[target_idx, wt_token_id].item()
                    )
                    total_llr += llr
                    mut_names.append(f"{wt_aa}{pos + 1}{mut_aa}")

                mut_str = ", ".join(mut_names)
                print(f"[{target_name}] [{header}] -> {mut_str} | LLR: {total_llr:.4f}")
                results.append({
                    "header": header,
                    "mutations": mut_str,
                    "total_llr": round(total_llr, 4)
                })

            except Exception as e:
                print(f"Error processing {header}: {e}")
                results.append({"header": header, "mutations": "ERROR", "total_llr": "ERROR"})

        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["header", "mutations", "total_llr"])
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ {target_name} 打分完成！结果已保存至: {output_csv}\n")

    print("🎉 ESMC-6B 所有批量任务已全部执行完毕！")


if __name__ == "__main__":
    main()