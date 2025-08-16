#!/usr/bin/env python3
"""
w2v_ft_embeddings.py
--------------------
Sequence-level embeddings from k-mers using Word2Vec / FastText → CSV.

- FASTA deflines: >ID|class   (class optional)
- DNA or Protein sequences
- Train a model or load a pretrained gensim model
- Pool k-mer vectors into a fixed-length vector (mean / sum / max / mean+max)
- DNA reverse-complement merge (strand-agnostic)
- CSV output: ID, features..., class
"""

import argparse
import csv
import gzip
import os
import sys
import random
from typing import List, Dict, Iterable, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

try:
    import numpy as np
    from gensim.models import Word2Vec, FastText, KeyedVectors
except Exception as e:
    print("[error] This script requires gensim and numpy. Try: pip install gensim tqdm", file=sys.stderr)
    sys.exit(1)

# ==========================
# Alphabets & helpers
# ==========================

DNA_ORDER = tuple("ACGT")
DNA_SET = set(DNA_ORDER)
DNA_COMP = {"A":"T","C":"G","G":"C","T":"A"}

AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

# Ambiguity (used only for filtering when ambig=skip or allowing when ambig=keep)
DNA_VALID = DNA_SET
AA_VALID  = AA_SET.union({"B","Z","X"})  # allowed if ambig=keep

def reverse_complement(seq: str) -> str:
    return "".join(DNA_COMP.get(b, b) for b in reversed(seq))

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
    """
    >ID|class or >ID         → returns (ID or None, class or None)
    """
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
# Tokenization
# ==========================

def valid_set(seq_type: str):
    return DNA_VALID if seq_type == "dna" else AA_VALID

def tokenize_kmers(seq: str, k: int, stride: int=1, seq_type: str="dna", ambig: str="skip") -> List[str]:
    V = valid_set(seq_type)
    n = len(seq)
    out = []
    for i in range(0, n - k + 1, stride):
        tok = seq[i:i+k]
        if ambig == "skip":
            if seq_type == "dna":
                if set(tok) <= DNA_SET:
                    out.append(tok)
            else:
                if set(tok).issubset(AA_SET):
                    out.append(tok)
        else:  # keep ambiguous token as-is
            out.append(tok)
    return out

# ==========================
# Corpus iterator for training
# ==========================

class KMersCorpus:
    def __init__(self, records, k, stride, seq_type, ambig):
        self.records = records
        self.k = k
        self.stride = stride
        self.seq_type = seq_type
        self.ambig = ambig
    def __iter__(self):
        for rec in self.records:
            toks = tokenize_kmers(rec["seq"], self.k, self.stride, self.seq_type, self.ambig)
            if toks:
                yield toks

# ==========================
# Embedding helpers
# ==========================

def pool_vectors(vecs: np.ndarray, mode: str="mean") -> np.ndarray:
    if vecs.size == 0:
        return None
    if mode == "mean":
        return vecs.mean(axis=0)
    if mode == "sum":
        return vecs.sum(axis=0)
    if mode == "max":
        return vecs.max(axis=0)
    if mode == "mean+max":
        return np.concatenate([vecs.mean(axis=0), vecs.max(axis=0)], axis=0)
    return vecs.mean(axis=0)

def seq_embedding_from_model(seq: str, model, k: int, stride: int, seq_type: str,
                             ambig: str, pool: str="mean") -> np.ndarray:
    toks = tokenize_kmers(seq, k, stride, seq_type, ambig)
    vec_list = []
    # Gensim Word2Vec/FastText: use .wv for vocab vectors; FastText model (not .wv) can synthesize OOV
    wv = model.wv if hasattr(model, "wv") else model
    for t in toks:
        if t in wv.key_to_index:
            vec_list.append(wv.get_vector(t))
        else:
            # FastText full model can return OOV vectors
            if hasattr(model, "get_vector"):
                try:
                    vec_list.append(model.get_vector(t))
                except Exception:
                    pass
    if len(vec_list) == 0:
        dim = wv.vector_size
        return np.zeros(2*dim if pool == "mean+max" else dim, dtype=float)
    vecs = np.stack(vec_list, axis=0)
    return pool_vectors(vecs, pool)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

# ==========================
# CLI
# ==========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Word2Vec / FastText k-mer embeddings from DNA/Protein FASTA → CSV."
    )
    # IO
    p.add_argument("-i", "--input", nargs="+", required=True,
                   help="FASTA file(s) (.fa/.fasta[.gz]). Deflines: >ID|class (class optional).")
    p.add_argument("-o", "--output", required=True,
                   help="Output CSV file.")
    # Sequence type & tokenization
    p.add_argument("-t", "--type", choices=["dna","protein"], required=True,
                   help="Sequence type.")
    p.add_argument("-k", type=int, required=True,
                   help="k-mer length (e.g., 3).")
    p.add_argument("--stride", type=int, default=1,
                   help="Stride between consecutive k-mers (default: 1).")
    p.add_argument("--ambig", choices=["skip","keep"], default="skip",
                   help="Ambiguous letters: 'skip' drops tokens containing them; 'keep' keeps them verbatim.")
    # Model: train or load
    p.add_argument("--algo", choices=["word2vec","fasttext"], default="word2vec",
                   help="Algorithm for training (or to match the pretrained model).")
    p.add_argument("--pretrained", type=str,
                   help="Path to a pretrained gensim model (.model/.bin/etc). If set, training is skipped.")
    p.add_argument("--vector-size", type=int, default=100,
                   help="Embedding dimension when training.")
    p.add_argument("--window", type=int, default=5,
                   help="Context window size for training.")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs.")
    p.add_argument("--min-count", type=int, default=1,
                   help="Min token frequency to include in vocab (training).")
    p.add_argument("--sg", type=int, choices=[0,1], default=1,
                   help="Training architecture: 1=skip-gram, 0=CBOW (default: 1).")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed for training.")
    p.add_argument("--save-model", type=str,
                   help="Optional path to save the trained model.")
    p.add_argument("--save-vocab", type=str,
                   help="Optional path to save the learned token vocabulary (.txt).")
    # Embedding aggregation
    p.add_argument("--pool", choices=["mean","sum","max","mean+max"], default="mean",
                   help="Pooling across token vectors to get a sequence embedding. 'mean+max' concatenates both.")
    p.add_argument("--l2norm", action="store_true",
                   help="L2-normalize the final sequence embedding vector.")
    p.add_argument("--reverse-complement-merge", action="store_true",
                   help="DNA only: average the forward embedding with the reverse-complement embedding.")
    # Output column naming
    p.add_argument("--col-style", choices=["auto","simple"], default="auto",
                   help="Column names: 'auto' → EMB_MEAN_i/EMB_MAX_i for mean+max, EMB_i otherwise; "
                        "'simple' → always EMB_1..EMB_N.")
    # Misc behavior
    p.add_argument("--uppercase", action="store_true",
                   help="Uppercase sequences (and U→T for DNA).")
    p.add_argument("--minlen", type=int, default=0,
                   help="Skip sequences shorter than this after cleaning (default: 0).")
    p.add_argument("--add-length", action="store_true",
                   help="Append 'Seq_length' column.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count()-1),
                   help="Parallel workers for training (inference runs in-process).")
    p.add_argument("--progress", action="store_true",
                   help="Show a progress bar if tqdm is installed.")
    return p.parse_args()

# ==========================
# Training / Loading
# ==========================

def train_or_load_model(records, args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load pretrained model if provided
    if args.pretrained:
        # Try loading with Word2Vec/FastText .load
        try:
            return Word2Vec.load(args.pretrained)
        except Exception:
            pass
        try:
            return FastText.load(args.pretrained)
        except Exception:
            pass
        # Try loading KeyedVectors (vectors only)
        try:
            kv = KeyedVectors.load(args.pretrained, mmap='r')
            dummy = Word2Vec(vector_size=kv.vector_size, min_count=1)
            dummy.build_vocab_from_freq({w:1 for w in kv.key_to_index})
            dummy.wv = kv
            return dummy
        except Exception as e:
            print(f"[error] Could not load pretrained model: {e}", file=sys.stderr)
            sys.exit(1)

    # Train from scratch
    corpus = KMersCorpus(records, args.k, args.stride, args.type, args.ambig)
    sentences = list(corpus)  # for very large datasets, stream instead
    if len(sentences) == 0:
        print("[error] No tokens found to train embeddings. Check -k/--stride/--ambig.", file=sys.stderr)
        sys.exit(1)

    if args.algo == "word2vec":
        model = Word2Vec(
            sentences=sentences,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=max(1, args.workers),
            epochs=args.epochs,
            sg=args.sg,
            seed=args.seed
        )
    else:
        model = FastText(
            sentences=sentences,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=max(1, args.workers),
            epochs=args.epochs,
            sg=args.sg,
            seed=args.seed
        )

    if args.save_model:
        try:
            model.save(args.save_model)
            print(f"Saved model: {args.save_model}")
        except Exception as e:
            print(f"[warn] Could not save model: {e}", file=sys.stderr)

    if args.save_vocab:
        try:
            with open(args.save_vocab, "w", encoding="utf-8") as fh:
                for w in model.wv.key_to_index.keys():
                    fh.write(w + "\n")
            print(f"Saved vocab: {args.save_vocab}")
        except Exception as e:
            print(f"[warn] Could not save vocab: {e}", file=sys.stderr)

    return model

# ==========================
# Main
# ==========================

def main():
    args = parse_args()

    # Read sequences
    records = list(read_fasta(
        args.input, args.type,
        force_upper=args.uppercase or True,
        minlen=args.minlen
    ))
    if not records:
        print("No sequences passed filters.", file=sys.stderr)
        sys.exit(1)

    # Train or load model
    model = train_or_load_model(records, args)

    # Determine header names
    dim = model.wv.vector_size
    if args.pool == "mean+max":
        if args.col_style == "simple":  # (guard against typo; replaced below)
            pass
    # Proper column naming
    if args.pool == "mean+max":
        if args.col_style == "simple":
            feature_names = [f"EMB_{i+1}" for i in range(2*dim)]
        else:
            feature_names = [f"EMB_MEAN_{i+1}" for i in range(dim)] + \
                            [f"EMB_MAX_{i+1}"  for i in range(dim)]
    else:
        feature_names = [f"EMB_{i+1}" for i in range(dim)]

    if args.add_length:
        feature_names.append("Seq_length")
    header = ["ID"] + feature_names + ["class"]

    # Process sequences (single-process so the model stays in memory)
    results = []
    iterator = records
    if args.progress and HAS_TQDM:
        iterator = tqdm(records, total=len(records), desc="Embedding", unit="seq")
    for idx, rec in enumerate(iterator):
        emb = seq_embedding_from_model(
            rec["seq"], model,
            k=args.k, stride=args.stride,
            seq_type=args.type, ambig=args.ambig,
            pool=args.pool
        )
        if args.type == "dna" and args.reverse_complement_merge:
            emb_rc = seq_embedding_from_model(
                reverse_complement(rec["seq"]), model,
                k=args.k, stride=args.stride,
                seq_type=args.type, ambig=args.ambig,
                pool=args.pool
            )
            emb = 0.5*(emb + emb_rc)
        if args.l2norm:
            emb = l2_normalize(emb)
        feats = emb.astype(float)
        if args.add_length:
            feats = np.concatenate([feats, np.array([rec["length"]], dtype=float)])
        results.append((idx, rec["id"], rec["class"], feats))

    results.sort(key=lambda x: x[0])

    # Resolve IDs/classes & write CSV
    seen, serial = set(), 1
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for _, sid, sclass, vec in results:
            rid = sid if (sid and sid not in seen) else str(serial)
            if (sid is None) or (sid in seen):
                serial += 1
            seen.add(rid)
            rclass = sclass if sclass is not None else "0"
            row = [rid] + list(vec.tolist()) + [rclass]
            w.writerow(row)

    print(f"Saved: {args.output}")
    print(f"Sequences: {len(results)} | Feature columns (excluding ID/class): {len(feature_names)}")

if __name__ == "__main__":
    main()

"""
-i / --input
        One or more DNA/Protein FASTA files; .gz supported. Deflines: >ID|class (class optional).
-o / --output
        Output CSV path.
-t / --type {dna,protein}
        Choose alphabet.
-k
        k-mer length (e.g., 3 for trimers).
--stride
        Step between consecutive k-mers (default 1).
--ambig {skip,keep}
        Ambiguity handling for tokens:
        skip: drop any k-mer containing ambiguous letters.
        keep: keep tokens verbatim (model can learn them).
--algo {word2vec,fasttext}
        Algorithm if training. FastText can synthesize vectors for unseen tokens via subwords.
--pretrained PATH
        Load a pretrained gensim model (.model, .bin, or a saved KeyedVectors). Skips training.
        Training hyperparameters (used only if not --pretrained):
--vector-size (dim), --window, --epochs, --min-count, --sg {1,0}, --seed, --workers.
--save-model PATH
        Save the trained model (optional).
--save-vocab PATH
        Save the learned token vocabulary (one token per line).
--pool {mean,sum,max,mean+max}
        Pool k-mer vectors into a sequence embedding. mean+max concatenates the mean and max vectors (size doubles).
--l2norm
        L2-normalize the final sequence embedding.
--reverse-complement-merge
        DNA only: average the forward embedding with the reverse-complement embedding (strand-agnostic).
--col-style {auto,simple}
        Column naming:
        auto: EMB_MEAN_i + EMB_MAX_i when using mean+max, else EMB_i.
        simple: always EMB_1..EMB_N.
--uppercase
        Uppercase sequences (and convert U→T for DNA).
--minlen
        Skip sequences shorter than this after cleaning.
--add-length
        Append a Seq_length column to the CSV.
--progress
        Show a progress bar (requires tqdm).


Examples:
# A) Train Word2Vec on DNA 3-mers, mean pooling, 128-dim; save model & vocab
python3 w2v_ft_embeddings.py -i dna.fa.gz -o dna_w2v_mean.csv -t dna -k 3 \
  --algo word2vec --vector-size 128 --window 5 --epochs 10 --sg 1 \
  --save-model dna_w2v_k3.model --save-vocab dna_w2v_k3_vocab.txt --progress

# B) Train Word2Vec on DNA 6-mers, mean+max pooling (concat), RC-merge, L2 normalize
python3 w2v_ft_embeddings.py -i dna.fa -o dna_w2v_meanmax_rc.csv -t dna -k 6 \
  --algo word2vec --vector-size 200 --window 4 --epochs 8 --sg 1 \
  --pool mean+max --reverse-complement-merge --l2norm --progress

# C) Train Word2Vec on DNA 5-mers, stride=2 (fewer tokens), max pooling
python3 w2v_ft_embeddings.py -i genomes.fasta -o dna_w2v_stride2_max.csv -t dna -k 5 \
  --stride 2 --algo word2vec --vector-size 100 --window 5 --epochs 6 --sg 1 \
  --pool max --progress

# D) Train Word2Vec on DNA 4-mers, sum pooling, skip ambiguous tokens, min length filter
python3 w2v_ft_embeddings.py -i contigs.fa.gz -o dna_w2v_sum.csv -t dna -k 4 \
  --algo word2vec --vector-size 150 --window 5 --epochs 12 --sg 1 \
  --pool sum --ambig skip --minlen 100 --progress

# E) Train Word2Vec on DNA 3-mers, mean pooling, simple column names, also save vocab
python3 w2v_ft_embeddings.py -i promoters.fa -o dna_w2v_mean_simple.csv -t dna -k 3 \
  --algo word2vec --vector-size 128 --window 5 --epochs 10 --sg 1 \
  --pool mean --col-style simple --save-vocab dna_w2v_k3_vocab.txt --progress





################################################################################
Alright — here’s how mean + max pooling concatenation works with the same tiny example.

Example sequence
ACGT
Step 1 — Break into k-mers
For k = 2:

AC, CG, GT
Step 2 — Look up embedding vectors
Assume a 3-dimensional embedding:

k-mer	EMB_1	EMB_2	EMB_3
AC	    0.2	    -0.1	0.4
CG	    0.0	    0.3	    -0.2
GT	    -0.1	0.2	    0.5

Step 3 — Mean pooling
Average each column:

Mean_1 = (0.2 + 0.0 + -0.1) / 3 = 0.0333
Mean_2 = (-0.1 + 0.3 + 0.2) / 3 = 0.1333
Mean_3 = (0.4 + -0.2 + 0.5) / 3 = 0.2333
Step 4 — Max pooling
Take the maximum value for each column:

Max_1 = max(0.2, 0.0, -0.1) = 0.2
Max_2 = max(-0.1, 0.3, 0.2) = 0.3
Max_3 = max(0.4, -0.2, 0.5) = 0.5
Step 5 — Concatenate mean + max
We put them end to end:

Final vector = [Mean_1, Mean_2, Mean_3, Max_1, Max_2, Max_3]
              = [0.0333, 0.1333, 0.2333, 0.2, 0.3, 0.5]
Step 6 — Save to CSV
Row format:

ID, EMB_1, EMB_2, EMB_3, EMB_4, EMB_5, EMB_6, class
Seq1, 0.0333, 0.1333, 0.2333, 0.2, 0.3, 0.5, 0

"""
