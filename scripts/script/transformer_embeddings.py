#!/usr/bin/env python3
"""
transformer_embeddings.py
-------------------------
Transformer-based embeddings for DNA/Protein FASTA → CSV.

Methods:
  - protbert : 'Rostlab/prot_bert_bfd'
  - esm2     : 'facebook/esm2_t6_8M_UR50D' (HF port; change via --model)
  - dnabert  : one of the DNABERT k-mer models (default 'zhihan1996/DNABERT-6'); requires --k-mer
  - auto     : user-specified --model with AutoTokenizer/AutoModel

Key features:
  - CPU/GPU selectable (no Triton)
  - Pooling: cls | mean | mean+max
  - Reverse-complement merge for DNA
  - Robust FASTA parsing (>ID|class), CSV output

Example:
  CPU (ProtBERT): 
    python3 transformer_embeddings.py -i prots.fa -o prot_bert_mean.csv -t protein \
      --method protbert --pool mean --batch-size 8 --device cpu --progress

  GPU (DNABERT 6-mer, mean+max, RC-merge):
    python3 transformer_embeddings.py -i dna.fa -o dna_dnabert_mm.csv -t dna \
      --method dnabert --k-mer 6 --pool mean+max --reverse-complement-merge \
      --batch-size 4 --max-length 512 --device gpu --progress
"""

import argparse
import csv
import gzip
import os
import sys
from typing import List, Tuple

# --- Dependencies (no Triton needed) ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    print("[error] This script requires: pip install torch transformers", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# ----------------------------
# FASTA & basic sequence utils
# ----------------------------

DNA_COMP = {"A":"T","C":"G","G":"C","T":"A","U":"T",
            "a":"T","c":"G","g":"C","t":"A","u":"T"}

def reverse_complement(seq: str) -> str:
    return "".join(DNA_COMP.get(b, "N") for b in seq[::-1])

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline=None)
    return open(path, "r", encoding="utf-8")

def sanitize_seq(seq: str, seq_type: str, force_upper: bool=True) -> str:
    s = seq.replace(" ", "").replace("\n", "").replace("\r", "")
    if force_upper:
        s = s.upper()
    if seq_type == "dna":
        s = s.replace("U", "T")
    return s

def parse_defline(defline: str) -> Tuple[str, str]:
    """
    >ID|class or >ID → (ID or None, class or None)
    """
    d = defline[1:].strip()
    first = d.split()[0] if d else ""
    if "|" in first:
        a, b = first.split("|", 1)
        return (a.strip() or None, b.strip() or None)
    return (first.strip() or None, None)

def read_fasta(paths: List[str], seq_type: str, minlen: int=0):
    cur_def, cur_seq = None, []
    for path in paths:
        with open_maybe_gzip(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                if line.startswith(">"):
                    if cur_def is not None:
                        sid, sclass = parse_defline(cur_def)
                        seq = sanitize_seq("".join(cur_seq), seq_type, True)
                        if len(seq) >= minlen:
                            yield {"id": sid, "class": sclass, "seq": seq, "length": len(seq)}
                    cur_def, cur_seq = line.rstrip("\n"), []
                else:
                    cur_seq.append(line.strip())
    if cur_def is not None:
        sid, sclass = parse_defline(cur_def)
        seq = sanitize_seq("".join(cur_seq), seq_type, True)
        if len(seq) >= minlen:
            yield {"id": sid, "class": sclass, "seq": seq, "length": len(seq)}

# ----------------------------
# Method-specific preprocessing
# ----------------------------

def space_separate_residues(seq: str) -> str:
    # For ProtBERT/ESM2 (HF ports), tokens are single residues separated by spaces.
    return " ".join(list(seq))

def make_dnabert_kmers(seq: str, k: int) -> str:
    # DNABERT expects space-separated k-mers (A/C/G/T only).
    seq = seq.replace("U", "T")
    seq = "".join(ch for ch in seq if ch in "ACGT")
    if len(seq) < k:
        return ""  # will yield an empty tokenization; handle upstream
    toks = [seq[i:i+k] for i in range(len(seq)-k+1)]
    return " ".join(toks)

# ----------------------------
# Pooling helpers
# ----------------------------

def pool_hidden(last_hidden: torch.Tensor, attn_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """
    last_hidden: [B, T, H]; attn_mask: [B, T] (1 for real tokens, 0 for pad/specials)
    mode: 'cls' uses token 0; 'mean' masks average; 'mean+max' concatenates mean and max.
    """
    if mode == "cls":
        return last_hidden[:, 0, :]  # [B, H]
    # mean / max ignore padded positions via attention_mask
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
    masked = last_hidden * mask  # zero out pads
    lengths = mask.sum(dim=1).clamp(min=1.0)            # [B,1]
    mean_vec = masked.sum(dim=1) / lengths              # [B,H]
    if mode == "mean":
        return mean_vec
    # max over valid tokens: set pads to very negative before max
    neg_inf = torch.finfo(last_hidden.dtype).min
    #------------------
    if last_hidden.size(2) == attn_mask.size(1):  # hidden is [B,H,T]
       last_hidden = last_hidden.transpose(1, 2) # -> [B,T,H]
    masked_for_max = last_hidden.masked_fill(attn_mask.unsqueeze(-1).eq(0), neg_inf)
    #------------------
    #masked_for_max = last_hidden.masked_fill(attn_mask==0, neg_inf)
    max_vec = masked_for_max.max(dim=1).values
    if mode == "max":
        return max_vec
    if mode == "mean+max":
        return torch.cat([mean_vec, max_vec], dim=-1)
    return mean_vec

def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)

# ----------------------------
# Model registry
# ----------------------------

DEFAULTS = {
    "protbert": "Rostlab/prot_bert_bfd",
    "esm2":     "facebook/esm2_t6_8M_UR50D",
    "dnabert":  "zhihan1996/DNABERT-6",   # requires --k-mer matching this model (e.g., 6)
    "auto":     None,                      # require --model
}

# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Transformer-based embeddings (ProtBERT / ESM2 / DNABERT / custom) → CSV."
    )
    # IO / basics
    p.add_argument("-i", "--input", nargs="+", required=True,
                   help="FASTA file(s) (.fa/.fasta[.gz]). Deflines: >ID|class (class optional).")
    p.add_argument("-o", "--output", required=True,
                   help="Output CSV file.")
    p.add_argument("-t", "--type", choices=["dna","protein"], required=True,
                   help="Sequence type.")
    p.add_argument("--method", choices=["protbert","esm2","dnabert","auto"], default="protbert",
                   help="Embedding method (defaults to protbert).")
    p.add_argument("--model", type=str,
                   help="Override HF model name or path (required when --method auto).")
    # Tokenization / shapes
    p.add_argument("--k-mer", type=int,
                   help="For DNABERT: k-mer length (e.g., 3..6). Must match the chosen DNABERT checkpoint.")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Max tokenized length (truncation/padding). Includes special tokens.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for inference.")
    p.add_argument("--pool", choices=["cls","mean","mean+max"], default="mean",
                   help="Pooling strategy over token embeddings.")
    # Device & runtime
    p.add_argument("--device", choices=["cpu","gpu"], default="cpu",
                   help="Use CPU or GPU. If GPU is chosen but unavailable, falls back to CPU.")
    p.add_argument("--l2norm", action="store_true",
                   help="L2-normalize the final embeddings.")
    p.add_argument("--reverse-complement-merge", action="store_true",
                   help="DNA only: average forward & reverse-complement embeddings.")
    # Filters & extras
    p.add_argument("--minlen", type=int, default=0,
                   help="Skip sequences shorter than this after cleaning.")
    p.add_argument("--add-length", action="store_true",
                   help="Append 'Seq_length' column.")
    p.add_argument("--progress", action="store_true",
                   help="Show a progress bar if tqdm is installed.")
    return p.parse_args()

def resolve_device(choice: str) -> torch.device:
    if choice == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "gpu" and not torch.cuda.is_available():
        print("[warn] CUDA not available; using CPU.", file=sys.stderr)
    return torch.device("cpu")

def prepare_inputs(texts: List[str], tokenizer: AutoTokenizer, max_len: int, device: torch.device):
    # Tokenize with truncation/padding to fixed length
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

def build_text(seq: str, seq_type: str, method: str, kmer: int=None) -> str:
    if seq_type == "protein":
        # ProtBERT/ESM2 usually expect space-separated residues
        return space_separate_residues(seq)
    # DNA
    if method == "dnabert":
        if not kmer:
            raise ValueError("DNABERT requires --k-mer.")
        km = make_dnabert_kmers(seq, kmer)
        return km
    # For generic DNA with non-DNABERT models, pass raw sequence (tokenizer-dependent).
    return seq

def embed_batch(texts: List[str], tokenizer, model, pool_mode: str, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        inputs = prepare_inputs(texts, tokenizer, max_len=args.max_length, device=device)
        outputs = model(**inputs)
        # Most encoder models expose last_hidden_state
        last_hidden = outputs.last_hidden_state  # [B,T,H]
        pooled = pool_hidden(last_hidden, inputs["attention_mask"], pool_mode)  # [B,H] or [B,2H]
    return pooled

if __name__ == "__main__":
    args = parse_args()

    # Resolve model name
    if args.method == "auto":
        if not args.model:
            print("[error] --model is required when --method auto", file=sys.stderr)
            sys.exit(1)
        model_name = args.model
    else:
        model_name = args.model or DEFAULTS[args.method]

    # Safety check for DNABERT
    if args.method == "dnabert" and not args.k_mer:
        # allow both --k-mer and --k_mer variants just in case
        pass

    # Device
    device = resolve_device(args.device)

    # Load tokenizer/model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        # some bio models lack fast tokenizers
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)

    # Read records
    records = list(read_fasta(args.input, args.type, minlen=args.minlen))
    if not records:
        print("No sequences passed filters.", file=sys.stderr)
        sys.exit(1)

    # Resolve IDs/classes
    seen, serial = set(), 1
    ids, classes, seqs, lengths = [], [], [], []
    for rec in records:
        sid = rec["id"]
        rid = sid if (sid and sid not in seen) else str(serial)
        if (sid is None) or (sid in seen):
            serial += 1
        seen.add(rid)
        ids.append(rid)
        classes.append(rec["class"] if rec["class"] is not None else "0")
        seqs.append(rec["seq"])
        lengths.append(rec["length"])

    # Build model-specific texts (forward and optional RC)
    texts_fwd = []
    texts_rc  = [] if (args.type=="dna" and args.reverse_complement_merge) else None
    for s in seqs:
        texts_fwd.append(build_text(s, args.type, args.method, kmer=args.k_mer if hasattr(args,"k_mer") else args.k_mer if hasattr(args,"k_mer") else args.k_mer))
        if texts_rc is not None:
            rc = reverse_complement(s)
            texts_rc.append(build_text(rc, args.type, args.method, kmer=args.k_mer if hasattr(args,"k_mer") else args.k_mer))

    # Batch over texts
    B = args.batch_size
    vecs = []
    iterator = range(0, len(texts_fwd), B)
    if args.progress and HAS_TQDM:
        iterator = tqdm(iterator, total=(len(texts_fwd)+B-1)//B, desc="Embedding", unit="batch")

    for start in iterator:
        end = min(start+B, len(texts_fwd))
        batch_texts = texts_fwd[start:end]
        emb = embed_batch(batch_texts, tokenizer, model, args.pool, device)  # [b, H or 2H]

        if texts_rc is not None:
            rc_emb = embed_batch(texts_rc[start:end], tokenizer, model, args.pool, device)
            emb = 0.5*(emb + rc_emb)

        if args.l2norm:
            emb = l2_normalize(emb)

        vecs.append(emb.cpu())

    vecs = torch.cat(vecs, dim=0)  # [N, D]
    dim = vecs.shape[1]

    # Column names
    if args.pool == "mean+max":
        # split into two halves of H each (assumes model hidden size H)
        H = dim // 2
        feature_names = [f"EMB_MEAN_{i+1}" for i in range(H)] + [f"EMB_MAX_{i+1}" for i in range(H)]
    else:
        feature_names = [f"EMB_{i+1}" for i in range(dim)]
    if args.add_length:
        feature_names.append("Seq_length")

    header = ["ID"] + feature_names + ["class"]

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, rid in enumerate(ids):
            row = [rid] + list(map(float, vecs[i].tolist()))
            if args.add_length:
                row.append(lengths[i])
            row.append(classes[i])
            w.writerow(row)

    print(f"Saved: {args.output}")
    print(f"Sequences: {len(ids)} | Feature columns (excluding ID/class): {len(feature_names)}")


"""
Option summary
-i / --input — FASTA file(s), .gz OK. Deflines: >ID|class (class optional).
-o / --output — Output CSV path.
-t / --type {dna,protein} — Choose sequence type.
--method {protbert,esm2,dnabert,auto}
protbert → Rostlab/prot_bert_bfd
esm2 → facebook/esm2_t6_8M_UR50D (you can override)
dnabert → zhihan1996/DNABERT-6 (requires --k-mer)
auto → use --model to specify any HF encoder model
--model — Override HF model name (required if --method auto).
--k-mer — For DNABERT only (3–6). Sequence is converted to space-separated k-mers.
--max-length — Tokenization max length (with truncation/padding).
--batch-size — Inference batch size.
--pool {cls,mean,mean+max} — Pooling over token embeddings (mean+max concatenates).
--device {cpu,gpu} — Use CPU or GPU. If gpu requested but not available, falls back to CPU.
--l2norm — L2-normalize final embedding(s).
--reverse-complement-merge — DNA only: average forward and reverse-complement embeddings.
--minlen — Skip sequences shorter than this (post-cleaning).
--add-length — Append Seq_length column to CSV.
--progress — Show a progress bar (needs tqdm).


Examples:
# 1) ProtBERT on proteins, mean pooling, CPU
python3 transformer_embeddings.py -i ../test_data/prot_test.fasta -o prot_bert_mean.csv -t protein \
  --method protbert --pool mean --batch-size 8 --device cpu --progress

# 2) ESM2 on proteins, CLS pooling, GPU
python3 transformer_embeddings.py -i ../test_data/prot_test.fasta -o prot_esm2_cls.csv -t protein \
  --method esm2 --pool cls --batch-size 8 --device gpu --progress

# 3) DNABERT 6-mer on DNA, mean+max pooling, RC-merge, GPU (fallbacks to CPU if unavailable)
 python transformer_embeddings.py -i ../test_data/cds_test.fasta -o dna_dnabert_mm.csv -t dna --method dnabert --k-mer 6 --pool mean+max --reverse-complement-merge --batch-size 4 --max-length 512 --device cpu --model zhihan1996/DNA_bert_6 --progress
 

# 4) Any HF encoder (auto), mean pooling
python transformer_embeddings.py -i ../test_data/prot_test.fasta -o auto_model_mean.csv -t protein \
  --method auto --model facebook/esm2_t6_8M_UR50D \
  --pool mean --batch-size 8 --device cpu


Note for Example 4:
--method auto will happily load any Hugging Face protein LM that works with AutoTokenizer/AutoModel. Here are good, drop-in model IDs you can use in place of your-org/your-protein-encoder:

Solid defaults (ESM-2 family)
-----------------------------
facebook/esm2_t6_8M_UR50D – very fast on CPU
facebook/esm2_t12_35M_UR50D – still CPU-friendly
facebook/esm2_t30_150M_UR50D – bigger, slower
facebook/esm2_t33_650M_UR50D – large; best quality, heavy on CPU (ESM-2 checkpoints are official and widely used.) 


ProtTrans classics
-----------------
Rostlab/prot_bert (and Rostlab/prot_bert_bfd) – BERT-style encoders; expects UPPERCASE amino acids and was trained with space-separated tokens (your script’s tokenizer usually handles this). 


Rostlab/prot_t5_xl_half_uniref50-enc – encoder-only ProtT5 (great for embeddings). Note: it’s fp16; on CPU you should cast to float32 (their card explicitly notes fp16 isn’t usable on CPU without .float()). 

Newer/open protein LMs
----------------------
chandar-lab/AMPLIFY_120M (or chandar-lab/AMPLIFY_350M) – efficient encoders; require trust_remote_code=True. Stage-1 models are not extended beyond 512 residues; the card shows 512→2048 across stages and uses Flash-Attention on GPU in their example. On pure CPU prefer the 120M size. 

RaphaelMourad/Mistral-Prot-v1-134M – compact Mistral-style protein LM; loads with standard Transformers. 

Older but still usable
----------------------
facebook/esm1b_t33_650M_UR50S (ESM-1b). Meta recommends ESM-2 instead for most tasks, but this works if you need 1b-style baselines.

"""
