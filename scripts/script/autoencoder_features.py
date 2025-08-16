#!/usr/bin/env python3
"""
autoencoder_features.py
-----------------------
Unsupervised Autoencoder features from DNA/Protein FASTA → CSV.

- FASTA deflines: >ID|class   (class optional; IDs auto if missing/duplicate; class defaults to 0)
- Encodings:
    1) k-mer bag-of-words (counts or normalized freq) with optional hashing to fixed dim
    2) fixed-length one-hot (DNA: A/C/G/T; Protein: 20 AAs), pad/truncate
- Train or load a PyTorch autoencoder; export the encoder latent as features.
- Optional DNA reverse-complement merging (average latent of fwd & RC).
"""

import argparse
import csv
import gzip
import math
import os
import sys
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


# ==========================
# FASTA I/O
# ==========================

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline=None)
    return open(path, "r", encoding="utf-8")

def sanitize_seq(seq: str, seq_type: str, force_upper: bool) -> str:
    s = seq.strip()
    if force_upper:
        s = s.upper()
    if seq_type == "dna":
        s = s.replace("U", "T")
    return "".join(s.split())

def parse_defline(defline: str) -> Tuple[str, str]:
    d = defline[1:].strip()
    tok = d.split()[0] if d else ""
    if "|" in tok:
        a, b = tok.split("|", 1)
        return (a.strip() or None, b.strip() or None)
    return (tok.strip() or None, None)

def read_fasta(paths: List[str], seq_type: str, force_upper: bool=True, minlen: int=0):
    """
    Yields dicts: {'id':..., 'class':..., 'seq':..., 'length': cleaned_length}
    """
    current_def, current_seq = None, []
    for path in paths:
        with open_maybe_gzip(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                if line.startswith(">"):
                    if current_def is not None:
                        sid, sclass = parse_defline(current_def)
                        seq = sanitize_seq("".join(current_seq), seq_type, force_upper)
                        if len(seq) >= minlen:
                            yield {"id": sid, "class": sclass, "seq": seq, "length": len(seq)}
                    current_def, current_seq = line.rstrip("\n"), []
                else:
                    current_seq.append(line.strip())
    if current_def is not None:
        sid, sclass = parse_defline(current_def)
        seq = sanitize_seq("".join(current_seq), seq_type, force_upper)
        if len(seq) >= minlen:
            yield {"id": sid, "class": sclass, "seq": seq, "length": len(seq)}


# ==========================
# Alphabets / RC / tokenization
# ==========================

DNA_ORDER = tuple("ACGT")
DNA_SET = set(DNA_ORDER)
DNA_COMP = {"A":"T","C":"G","G":"C","T":"A"}

AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

def reverse_complement(seq: str) -> str:
    return "".join(DNA_COMP.get(b, "N") for b in reversed(seq))

def kmer_tokens(seq: str, k: int, stride: int, seq_type: str, ambig: str) -> List[str]:
    n = len(seq)
    out = []
    if seq_type == "dna":
        for i in range(0, n - k + 1, stride):
            tok = seq[i:i+k]
            if ambig == "skip":
                if set(tok) <= DNA_SET:
                    out.append(tok)
            else:
                out.append(tok)
    else:
        for i in range(0, n - k + 1, stride):
            tok = seq[i:i+k]
            if ambig == "skip":
                if set(tok).issubset(AA_SET):
                    out.append(tok)
            else:
                out.append(tok)
    return out


# ==========================
# Vectorizers
# ==========================

def stable_hash(s: str) -> int:
    # deterministic 64-bit FNV-1a like hash
    h = 1469598103934665603
    for c in s.encode("utf-8"):
        h ^= c
        h *= 1099511628211
        h &= (1<<64)-1
    return h

def vec_kmer_bow(tokens: List[str], vocab: Dict[str, int]=None, hash_dim: int=None) -> np.ndarray:
    """
    If vocab is provided → dense vector of len(vocab).
    Else if hash_dim provided → hashed vector of size hash_dim.
    """
    if vocab is not None:
        v = np.zeros(len(vocab), dtype=np.float32)
        for t in tokens:
            idx = vocab.get(t)
            if idx is not None:
                v[idx] += 1.0
        return v
    elif hash_dim is not None and hash_dim > 0:
        v = np.zeros(hash_dim, dtype=np.float32)
        for t in tokens:
            h = stable_hash(t) % hash_dim
            v[h] += 1.0
        return v
    else:
        raise ValueError("Provide either a vocab or a positive hash_dim for k-mer BOW.")

def vec_onehot_fixed(seq: str, pad_len: int, seq_type: str) -> np.ndarray:
    """
    Flattened one-hot vector of shape (pad_len * alphabet_size).
    """
    alpha = DNA_ORDER if seq_type == "dna" else AA_ORDER
    alpha_index = {ch:i for i,ch in enumerate(alpha)}
    A = len(alpha)
    L = min(len(seq), pad_len)
    v = np.zeros((pad_len, A), dtype=np.float32)
    for i in range(L):
        ch = seq[i]
        j = alpha_index.get(ch)
        if j is not None:
            v[i, j] = 1.0
    return v.reshape(-1)


# ==========================
# Dataset
# ==========================

class SeqVecDataset(Dataset):
    def __init__(self, vecs: np.ndarray):
        self.vecs = vecs
    def __len__(self):
        return self.vecs.shape[0]
    def __getitem__(self, idx):
        x = self.vecs[idx]
        return torch.from_numpy(x)


# ==========================
# Autoencoder model
# ==========================

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim, act="relu", dropout=0.0, bn=False):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if bn: layers.append(nn.BatchNorm1d(b))
            layers.append(get_activation(act))
            if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class MLPDecoder(nn.Module):
    def __init__(self, out_dim, hidden_dims, latent_dim, act="relu", dropout=0.0, bn=False):
        super().__init__()
        dims = [latent_dim] + list(reversed(hidden_dims))
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if bn: layers.append(nn.BatchNorm1d(b))
            layers.append(get_activation(act))
            if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim, act="relu", dropout=0.0, bn=False):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden_dims, latent_dim, act, dropout, bn)
        self.decoder = MLPDecoder(in_dim, hidden_dims, latent_dim, act, dropout, bn)
    def forward(self, x):
        z = self.encoder(x)
        xr = self.decoder(z)
        return xr, z

def get_activation(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    return nn.ReLU()


# ==========================
# Training / Eval
# ==========================

def train_autoencoder(model, loader, epochs, lr, wd, device, progress=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        it = loader
        if progress and HAS_TQDM:
            it = tqdm(loader, desc=f"Train AE epoch {ep+1}/{epochs}", leave=False)
        model.train()
        for xb in it:
            xb = xb.to(device, dtype=torch.float32)
            xr, z = model(xb)
            loss = loss_fn(xr, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

def encode(model, vecs: np.ndarray, device, batch_size=256) -> np.ndarray:
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, vecs.shape[0], batch_size):
            xb = torch.from_numpy(vecs[i:i+batch_size]).to(device, dtype=torch.float32)
            z = model.encoder(xb)
            out.append(z.cpu().numpy())
    return np.vstack(out)


# ==========================
# CLI
# ==========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Autoencoder features (unsupervised) from DNA/Protein FASTA → CSV."
    )
    # IO
    p.add_argument("-i", "--input", nargs="+", required=True,
                   help="FASTA file(s) (.fa/.fasta[.gz]). Deflines: >ID|class (class optional).")
    p.add_argument("-o", "--output", required=True,
                   help="Output CSV file.")

    # Sequence type & cleaning
    p.add_argument("-t", "--type", choices=["dna","protein"], required=True,
                   help="Sequence type.")
    p.add_argument("--uppercase", action="store_true",
                   help="Uppercase sequences (and U→T for DNA).")
    p.add_argument("--minlen", type=int, default=0,
                   help="Skip sequences shorter than this after cleaning.")

    # Encoding
    p.add_argument("--encoding", choices=["kmer-bow","onehot-fixed"], default="kmer-bow",
                   help="Input encoding for the AE: 'kmer-bow' (bag of k-mers) or 'onehot-fixed' (pad/truncate).")
    # k-mer options
    p.add_argument("-k", type=int, default=3,
                   help="k-mer length for 'kmer-bow'.")
    p.add_argument("--stride", type=int, default=1,
                   help="Stride for k-mers.")
    p.add_argument("--ambig", choices=["skip","keep"], default="skip",
                   help="Ambiguity handling in k-mers.")
    p.add_argument("--normalize", choices=["none","l1","l2"], default="l1",
                   help="Normalize input vectors per sequence (kmer-bow/onehot).")
    p.add_argument("--hash-dim", type=int, default=8192,
                   help="If >0, hash k-mers into this many bins (recommended for large vocab). If 0, build explicit vocab.")
    p.add_argument("--save-vocab", type=str,
                   help="Save explicit k-mer vocab (one k-mer per line). Used only if --hash-dim 0.")
    # onehot options
    p.add_argument("--pad-len", type=int, default=512,
                   help="Sequence length for one-hot (pad/truncate).")

    # DNA reverse complement
    p.add_argument("--reverse-complement-merge", action="store_true",
                   help="DNA only: average latent of forward and reverse-complement encodings.")

    # Autoencoder architecture
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 256],
                   help="Hidden layer sizes for the encoder/decoder (mirrored).")
    p.add_argument("--latent-dim", type=int, default=128,
                   help="Latent dimension to export as features.")
    p.add_argument("--act", choices=["relu","gelu","tanh"], default="relu",
                   help="Activation function.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate (0..1).")
    p.add_argument("--batch-norm", action="store_true",
                   help="Use BatchNorm between linear layers.")

    # Training
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay (AdamW).")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto",
                   help="Device to use (auto selects CUDA if available).")
    p.add_argument("--progress", action="store_true",
                   help="Show training/progress bars if tqdm is installed.")

    # Persistence / postproc
    p.add_argument("--pretrained", type=str,
                   help="Path to a saved AE .pt model to load (skips training).")
    p.add_argument("--save-model", type=str,
                   help="Path to save the trained AE .pt model.")
    p.add_argument("--l2norm", action="store_true",
                   help="L2-normalize the latent vector before writing CSV.")
    p.add_argument("--add-length", action="store_true",
                   help="Append 'Seq_length' column.")
    return p.parse_args()


# ==========================
# Utils
# ==========================

def maybe_norm(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "l1":
        s = np.sum(np.abs(x))
        return x if s == 0 else (x / s)
    if mode == "l2":
        n = np.linalg.norm(x)
        return x if n == 0 else (x / n)
    return x

def build_vocab(all_tokens: List[List[str]]) -> Dict[str, int]:
    # Build vocab from all tokens (only if --hash-dim 0)
    counter = Counter()
    for toks in all_tokens:
        counter.update(toks)
    vocab = {tok:i for i, (tok, _) in enumerate(sorted(counter.items()))}
    return vocab


# ==========================
# Main
# ==========================

def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))

    # Read sequences
    records = list(read_fasta(
        args.input, args.type,
        force_upper=args.uppercase or True,
        minlen=args.minlen
    ))
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

    # ----- Build input vectors -----
    if args.encoding == "kmer-bow":
        # Tokenize
        all_tokens = [kmer_tokens(s, args.k, args.stride, args.type, args.ambig) for s in seqs]

        if args.hash_dim == 0:  # guard for typo; kept for safety in case of paste errors
            pass
        if args.hash_dim and args.hash_dim > 0:
            vocab = None
            X = np.vstack([maybe_norm(vec_kmer_bow(toks, vocab=None, hash_dim=args.hash_dim), args.normalize)
                           for toks in all_tokens]).astype(np.float32)
            in_dim = X.shape[1]
        else:
            vocab = build_vocab(all_tokens)
            X = np.vstack([maybe_norm(vec_kmer_bow(toks, vocab=vocab), args.normalize)
                           for toks in all_tokens]).astype(np.float32)
            in_dim = X.shape[1]
            if args.save_vocab:
                with open(args.save_vocab, "w", encoding="utf-8") as fh:
                    for kmer, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
                        fh.write(f"{kmer}\n")
        encode_fn = lambda s: maybe_norm(
            vec_kmer_bow(kmer_tokens(s, args.k, args.stride, args.type, args.ambig),
                         vocab=vocab, hash_dim=(args.hash_dim if vocab is None else None)),
            args.normalize
        )

    else:  # onehot-fixed
        in_dim = (args.pad_len * (4 if args.type == "dna" else 20))
        X = np.vstack([maybe_norm(vec_onehot_fixed(s, args.pad_len, args.type), args.normalize)
                       for s in seqs]).astype(np.float32)
        encode_fn = lambda s: maybe_norm(vec_onehot_fixed(s, args.pad_len, args.type), args.normalize)

    # ----- Build/Load model -----
    ae = AutoEncoder(
        in_dim=in_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        act=args.act,
        dropout=args.dropout,
        bn=args.batch_norm
    )

    if args.pretrained and os.path.isfile(args.pretrained):
        ae.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
    else:
        ds = SeqVecDataset(X)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        train_autoencoder(ae, dl, epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                          device=device, progress=args.progress)
        if args.save_model:
            try:
                torch.save(ae.state_dict(), args.save_model)
                print(f"Saved model: {args.save_model}")
            except Exception as e:
                print(f"[warn] Could not save model: {e}", file=sys.stderr)

    # ----- Encode all sequences (with optional RC-merge) -----
    # Base encoding for all (already have X for forward)
    Z_fwd = encode(ae, X, device=device, batch_size=max(64, args.batch_size))

    if args.type == "dna" and args.reverse_complement_merge:
        # Build RC input vectors sequence-by-sequence (to avoid storing all at once)
        Z = []
        for s in (tqdm(seqs, desc="RC merge") if (args.progress and HAS_TQDM) else seqs):
            s_rc = reverse_complement(s)
            x_rc = encode_fn(s_rc).astype(np.float32)[None, :]
            z_rc = encode(ae, x_rc, device=device, batch_size=1)[0]
            Z.append(0.5 * (z_rc + Z_fwd[len(Z)]))
        Z = np.vstack(Z).astype(np.float32)
    else:
        Z = Z_fwd.astype(np.float32)

    if args.l2norm:
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Z = Z / norms

    # ----- Write CSV -----
    feature_names = [f"EMB_{i+1}" for i in range(args.latent_dim)]
    if args.add_length:
        feature_names.append("Seq_length")
    header = ["ID"] + feature_names + ["class"]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for rid, rclass, L, z in zip(ids, classes, lengths, Z):
            row = [rid] + list(z.astype(float))
            if args.add_length:
                row.append(L)
            row.append(rclass)
            w.writerow(row)

    print(f"Saved: {args.output}")
    print(f"Sequences: {len(ids)} | Feature columns (excluding ID/class): {args.latent_dim}")


if __name__ == "__main__":
    main()
"""
Option descriptions (quick)
--------------------------
-i / --input    — FASTA file(s); .gz supported; deflines >ID|class (class optional)
-o / --output   — Output CSV path
-t / --type     {dna,protein} — Alphabet (enables DNA RC merge)
--uppercase     — Uppercase sequences; DNA U→T
--minlen        — Skip sequences shorter than this after cleaning

Encoding
+++++++
--encoding  {kmer-bow,onehot-fixed} — Choose input to the autoencoder
            kmer-bow
-k — k-mer  length (default 3)
--stride    — step between k-mers (default 1)
--ambig     {skip,keep} — drop vs keep ambiguous tokens
--hash-dim  — if >0, hash k-mers to this dimension (default 8192, recommended)
--save-vocab — if --hash-dim 0, save explicit k-mer vocabulary

onehot-fixed
++++++++++++

--pad-len   — fixed length (pad/truncate); feature size = pad_len × 4 (DNA) or × 20 (protein)
--normalize {none,l1,l2} — Per-sequence normalization of input vector (default l1)

DNA
+++
--reverse-complement-merge — Average latent of forward & RC encodings (strand-agnostic)

Autoencoder
++++++++++
--hidden-dims — Encoder/decoder hidden sizes (mirrored), e.g. --hidden-dims 1024 256
--latent-dim — Size of exported latent vector (default 128)
--act {relu,gelu,tanh} — Activation
--dropout — Dropout probability (default 0.0)
--batch-norm — Add BatchNorm layers

Training
++++++++
--epochs — Training epochs (default 10)
--batch-size — Training batch size (default 256)
--lr — Learning rate (default 1e-3)
--weight-decay — AdamW weight decay (default 1e-4)
--device {auto,cpu,cuda} — Compute device (auto uses CUDA if available)
--progress — Show progress bars (needs tqdm)

Persistence & output
++++++++++++++++++++
--pretrained PATH — Load a saved .pt model (skips training)
--save-model PATH — Save the trained model
--l2norm — L2-normalize the latent feature vector
--add-length — Append Seq_length column


Examples:
# 1) DNA: k-mer bag (k=6) hashed to 16k dims → latent 256; RC-merge; save model
python autoencoder_features.py \
  -i dna.fa -o dna_ae_k6_lat256.csv -t dna \
  --encoding kmer-bow -k 6 --stride 1 --ambig skip --hash-dim 16384 --normalize l1 \
  --hidden-dims 2048 512 --latent-dim 256 --act relu --dropout 0.1 --batch-norm \
  --epochs 15 --batch-size 512 --lr 1e-3 --weight-decay 1e-4 \
  --reverse-complement-merge --l2norm --save-model dna_k6_ae.pt --progress

# 2) Protein: one-hot fixed length 512 → latent 128
python autoencoder_features.py \
  -i prots.fasta -o prot_ae_onehot512.csv -t protein \
  --encoding onehot-fixed --pad-len 512 --normalize l2 \
  --hidden-dims 4096 512 --latent-dim 128 --act gelu \
  --epochs 10 --batch-size 256 --lr 1e-3 --progress

# 3) DNA: load a pretrained model and just export features
python autoencoder_features.py \
  -i promoters.fa.gz -o promoters_ae.csv -t dna \
  --encoding kmer-bow -k 5 --hash-dim 8192 --normalize l1 \
  --latent-dim 128 --pretrained dna_k5_ae.pt --l2norm --progress


Notes
Prefer hashed k-mer vectors for big k (keeps memory bounded).
If you need a true vocabulary (for interpretability), set --hash-dim 0 (slower; larger vectors).
--reverse-complement-merge works with either encoding since it re-encodes RC before passing to the encoder.
"""
