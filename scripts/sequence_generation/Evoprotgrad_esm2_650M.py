import os
import torch
import pandas as pd
import evo_prot_grad
from transformers import AutoTokenizer, EsmForMaskedLM

def main():
    print("正在初始化环境并加载模型，请稍候...")
    
    # 1. 硬件配置：自动检测并分配至你之前的 cuda:1 显卡
    target_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    # 2. 准备野生型序列 (avGFP) 并生成临时 FASTA 文件
    wt_sequence = "MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR"
    temp_fasta_path = "wt_gfp.fasta"
    with open(temp_fasta_path, "w") as f:
        f.write(f">wildtype\n{wt_sequence}\n")
        
    # 3. 请出第一位专家：ESM-2 (650M 甜品级高精度版本，负责把控蛋白质折叠与结构)
    hf_model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"   -> 正在加载 ESM-2 结构专家 ({hf_model_name})...")
    
    my_model = EsmForMaskedLM.from_pretrained(hf_model_name)
    my_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    
    esm2_expert = evo_prot_grad.get_expert(
        'esm', 
        'mutant_marginal', 
        model=my_model,          
        tokenizer=my_tokenizer,  
        temperature=1.0, 
        device=target_device  
    )

    # 4. 请出第二位专家：NREL GFP 亮度预测模型 (负责把控荧光强度)
    print("   -> 正在加载 avGFP 荧光质检专家...")
    gfp_expert = evo_prot_grad.get_expert(
        'onehot_downstream_regression', 
        model='NREL/avGFP-fluorescence-onehot-cnn', 
        temperature=1.0, 
        device=target_device
    )

    # 5. 启动 MCMC 定向进化引擎
    print("模型加载完毕，开始双专家联合定向进化...")
    variants, scores = evo_prot_grad.DirectedEvolution(
        wt_fasta=temp_fasta_path,    
        output='all',                
        experts=[esm2_expert, gfp_expert], 
        parallel_chains=3,           # 并行探索 3 条路线
        n_steps=5,                   # 进化步数 (若需跑更久，可修改此数值)
        max_mutations=15,            # 每条序列最多允许积累 15 个突变
        verbose=False                
    )()

    # 6. 打包并保存结果为 CSV 表格
    output_csv = "esm2_8M&avGFP-fluorescence-onehot-cnn_cgregfp_evolution_results_650M.csv"
    df = pd.DataFrame({
        'Sequence': variants, 
        'Score': [score.item() if torch.is_tensor(score) else score for score in scores]
    })
    df.to_csv(output_csv, index=False)
    
    print(f"进化任务完成，共生成 {len(variants)} 个突变体。")
    print(f"结果已保存至本目录下的: {output_csv}")

if __name__ == "__main__":
    main()
