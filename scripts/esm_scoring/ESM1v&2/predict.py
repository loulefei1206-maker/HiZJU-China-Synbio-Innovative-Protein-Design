# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import string

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
    
    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""

    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )

    # fmt: off
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    # === 新增的 FASTA 输入参数 ===
    parser.add_argument(
        "--fasta-input",
        type=str,
        default=None,
        help="Path to input FASTA file containing mutant sequences"
    )
    # =============================
    parser.add_argument(
        "--dms-input",
        type=pathlib.Path,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=0,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA in a3m format (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to select from the start of the MSA"
    )
    # fmt: on
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def label_row(row, sequence, token_probs, alphabet, offset_idx):
    # === 升级版：兼容多位点突变 (如 A24V:K25R) 和野生型 (WT) ===
    if row == "WT":
        return 0.0
        
    total_score = 0.0
    mutations = row.split(':')
    
    for mut in mutations:
        wt, idx_str, mt = mut[0], mut[1:-1], mut[-1]
        idx = int(idx_str) - offset_idx
        assert sequence[idx] == wt, f"The listed wildtype ({wt}) does not match the provided sequence ({sequence[idx]}) at position {idx + offset_idx}"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        total_score += score.item()
        
    return total_score
    # =========================================================


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


def generate_mutations(sequence):
    mutations = []
    for i in range(len(sequence)):
        for aa in amino_acids:
            mut = sequence[i] + str(i+1)+aa 
            mutations.append(mut)
    return mutations


def main(args):
    # === 新增的 FASTA 解析与比对逻辑 ===
    if getattr(args, 'fasta_input', None) is not None:
        print(f"正在读取 FASTA 文件: {args.fasta_input}")
        wt_seq = args.sequence
        data = []
        for record in SeqIO.parse(args.fasta_input, "fasta"):
            mut_seq = str(record.seq).upper()
            
            # 严谨性校验：ESM zero-shot 仅支持替换突变，不支持插入/缺失
            if len(mut_seq) != len(wt_seq):
                print(f"警告: 序列 {record.id} 长度({len(mut_seq)})与野生型({len(wt_seq)})不符，已跳过。")
                continue
                
            muts = []
            # 逐位比对提取突变
            for i, (w, m) in enumerate(zip(wt_seq, mut_seq)):
                if w != m:
                    muts.append(f"{w}{i + args.offset_idx}{m}")
            
            if muts:
                # 多个突变以冒号分隔，序列ID单独保存一列方便溯源
                data.append({"sequence_id": record.id, args.mutation_col: ":".join(muts)})
            else:
                data.append({"sequence_id": record.id, args.mutation_col: "WT"})
                
        df = pd.DataFrame(data)
        print(f"成功从 FASTA 中解析了 {len(df)} 条有效突变序列。")
        
    elif getattr(args, 'dms_input', None) is not None:
        try:
            df = pd.read_csv(args.dms_input)
        except Exception as e:
            print(f"读取 DMS 输入文件失败: {e}")
            return
    else:
        # 保留老师原本的自动饱和突变功能作为备选
        df = pd.DataFrame(generate_mutations(args.sequence), columns=[args.mutation_col])
    # ====================================

    # inference for each model
    for model_location in args.model_location:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        if torch.cuda.is_available() and not args.nogpu:
            model = model.cuda()
            print("Transferred model to GPU")

        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):
            data = [read_msa(args.msa_path, args.msa_samples)]
            assert (
                args.scoring_strategy == "masked-marginals"
            ), "MSA Transformer only supports masked marginal strategy"

            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(2))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens_masked.cuda())["logits"], dim=-1
                    )
                all_token_probs.append(token_probs[:, 0, i])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            df[model_location] = df.apply(
                lambda row: label_row(
                    row[args.mutation_col], args.sequence, token_probs, alphabet, args.offset_idx
                ),
                axis=1,
            )

        else:
            data = [
                ("protein1", args.sequence),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            if args.scoring_strategy == "wt-marginals":
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[args.mutation_col],
                        args.sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif args.scoring_strategy == "masked-marginals":
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1))):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(
                            model(batch_tokens_masked.cuda())["logits"], dim=-1
                        )
                    all_token_probs.append(token_probs[:, i])  # vocab size
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[args.mutation_col],
                        args.sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif args.scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                df[model_location] = df.progress_apply(
                    lambda row: compute_pppl(
                        row[args.mutation_col], args.sequence, model, alphabet, args.offset_idx
                    ),
                    axis=1,
                )

    # 导出时去掉无用的索引列，保持结果干净
    df.to_csv(args.dms_output, float_format='%.3f', index=False)


if __name__ == "__main__":
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    parser = create_parser()
    args = parser.parse_args()
    main(args)