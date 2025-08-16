#!/usr/bin/env python3
"""
doc2vec_embeddings.py
---------------------
Sequence-level embeddings from k-mers using Doc2Vec → CSV.

- FASTA deflines: >ID|class   (class optional; IDs auto if missing/duplicate; class defaults to 0)
- DNA or Protein sequences
- Train Doc2Vec (PV-DM or PV-DBOW) or load a pretrained Doc2Vec model (.model)
- DNA reverse-complement merging (strand-agnostic)
- CSV output: ID, features..., class
"""

import argparse
import csv
import gzip
import os
import sys
import random
from typing import List, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

try:
    import numpy as np
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

# ambiguity filtering when ambig=skip (when ambig=keep we allow any symbol)
DNA_VALID = DNA_SET
AA_VALID  = AA_SET.union({"B","Z","X"})

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

def tokenize_kmers(seq: str, k: int, stride: int=1, seq_type: str="dna", ambig: str="skip") -> List[str]:
    n = len(seq)
    out = []
    if ambig == "skip":
        if seq_type == "dna":
            for i in range(0, n - k + 1, stride):
                tok = seq[i:i+k]
                if set(tok) <= DNA_SET:
                    out.append(tok)
        else:
            for i in range(0, n - k + 1, stride):
                tok = seq[i:i+k]
                if set(tok).issubset(AA_SET):
                    out.append(tok)
    else:  # keep
        for i in range(0, n - k + 1, stride):
            out.append(seq[i:i+k])
    return out

# ==========================
# Doc2Vec helpers
# ==========================

def build_tag(rid: str) -> str:
    # ensure tag is a string and unique across dataset
    return f"DOC__{rid}"

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

# ==========================
# CLI
# ==========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Doc2Vec k-mer embeddings from DNA/Protein FASTA → CSV."
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
    p.add_argument("--pretrained", type=str,
                   help="Path to a pretrained Doc2Vec .model. If set, training is skipped and vectors are inferred.")
    p.add_argument("--vector-size", type=int, default=100,
                   help="Embedding dimension when training.")
    p.add_argument("--window", type=int, default=5,
                   help="Context window size for PV-DM (ignored for PV-DBOW).")
    p.add_argument("--epochs", type=int, default=20,
                   help="Training epochs.")
    p.add_argument("--min-count", type=int, default=1,
                   help="Min token frequency to include in vocab.")
    p.add_argument("--dm", type=int, choices=[1,0], default=1,
                   help="Doc2Vec architecture: 1=PV-DM (default), 0=PV-DBOW.")
    p.add_argument("--dm-mean", type=int, choices=[1,0], default=1,
                   help="PV-DM only: 1=average context vectors (default), 0=concatenate if dm_concat=1.")
    p.add_argument("--dm-concat", type=int, choices=[1,0], default=0,
                   help="PV-DM only: 1=concatenate context vectors; implies larger effective vector space.")
    p.add_argument("--hs", type=int, choices=[1,0], default=0,
                   help="Hierarchical softmax (1) vs negative sampling (0).")
    p.add_argument("--negative", type=int, default=5,
                   help="Negative samples if hs=0 (default: 5).")
    p.add_argument("--sample", type=float, default=1e-3,
                   help="Downsample threshold for frequent tokens.")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count()-1),
                   help="Parallel workers for training.")
    p.add_argument("--save-model", type=str,
                   help="Optional path to save the trained Doc2Vec model.")
    # Inference / output
    p.add_argument("--infer-steps", type=int, default=50,
                   help="Infer steps per document when using pretrained model or forcing inference.")
    p.add_argument("--infer-alpha", type=float, default=0.025,
                   help="Initial alpha for inference.")
    p.add_argument("--force-infer", action="store_true",
                   help="Even if we just trained, infer vectors instead of reading trained docvecs (consistent with pretrained behavior).")
    p.add_argument("--l2norm", action="store_true",
                   help="L2-normalize the final sequence embedding vector.")
    p.add_argument("--reverse-complement-merge", action="store_true",
                   help="DNA only: average the forward doc vector with the reverse-complement doc vector.")
    # Misc behavior
    p.add_argument("--uppercase", action="store_true",
                   help="Uppercase sequences (and U→T for DNA).")
    p.add_argument("--minlen", type=int, default=0,
                   help="Skip sequences shorter than this after cleaning (default: 0).")
    p.add_argument("--add-length", action="store_true",
                   help="Append 'Seq_length' column.")
    p.add_argument("--progress", action="store_true",
                   help="Show a progress bar if tqdm is installed.")
    return p.parse_args()

# ==========================
# Training / Loading
# ==========================

def train_or_load_model(tagged_docs, args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.pretrained:
        try:
            model = Doc2Vec.load(args.pretrained)
            return model, True  # pretrained=True
        except Exception as e:
            print(f"[error] Could not load pretrained Doc2Vec model: {e}", file=sys.stderr)
            sys.exit(1)

    # Train from scratch
    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=max(1, args.workers),
        epochs=args.epochs,
        dm=args.dm,
        dm_mean=args.dm_mean,
        dm_concat=args.dm_concat,
        hs=args.hs,
        negative=args.negative,
        sample=args.sample,
        seed=args.seed,
    )

    if args.save_model:
        try:
            model.save(args.save_model)
            print(f"Saved model: {args.save_model}")
        except Exception as e:
            print(f"[warn] Could not save model: {e}", file=sys.stderr)

    return model, False  # pretrained=False

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

    # Resolve IDs & classes (ensure unique tags for Doc2Vec)
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

    # Build TaggedDocuments for training or for vocabulary alignment
    tagged_docs = []
    for rid, seq in zip(ids, seqs):
        toks = tokenize_kmers(seq, args.k, args.stride, args.type, args.ambig)
        tag = build_tag(rid)
        tagged_docs.append(TaggedDocument(words=toks, tags=[tag]))

    # Train or load model
    model, pretrained = train_or_load_model(tagged_docs, args)

    # Header
    dim = model.vector_size
    feature_names = [f"EMB_{i+1}" for i in range(dim)]
    if args.add_length:
        feature_names.append("Seq_length")
    header = ["ID"] + feature_names + ["class"]

    # Get vectors (infer or pull from trained docvecs)
    out_rows = []
    iterator = list(zip(ids, classes, seqs, lengths))
    if args.progress and HAS_TQDM:
        iterator = tqdm(iterator, total=len(ids), desc="Doc2Vec", unit="seq")

    for rid, rclass, seq, L in iterator:
        toks = tokenize_kmers(seq, args.k, args.stride, args.type, args.ambig)

        # Choose vector source
        if pretrained or args.force_infer:
            vec = model.infer_vector(toks, alpha=args.infer_alpha, epochs=args.infer_steps)
        else:
            # If we trained just now and didn't force inference, use the learned docvecs
            tag = build_tag(rid)
            if tag in model.dv:
                vec = model.dv[tag]
            else:
                # Fallback to inference if tag missing
                vec = model.infer_vector(toks, alpha=args.infer_alpha, epochs=args.infer_steps)

        # Reverse-complement merge (DNA only): average vectors of seq and RC(seq)
        if args.type == "dna" and args.reverse_complement_merge:
            toks_rc = tokenize_kmers(reverse_complement(seq), args.k, args.stride, args.type, args.ambig)
            vec_rc = model.infer_vector(toks_rc, alpha=args.infer_alpha, epochs=args.infer_steps)
            vec = 0.5 * (vec + vec_rc)

        if args.l2norm:
            vec = l2_normalize(vec)

        row = [rid] + list(np.asarray(vec, dtype=float))
        if args.add_length:
            row.append(L)
        row.append(rclass)
        out_rows.append(row)

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in out_rows:
            w.writerow(r)

    print(f"Saved: {args.output}")
    print(f"Sequences: {len(ids)} | Feature columns (excluding ID/class): {len(feature_names)}")

if __name__ == "__main__":
    main()

"""
-i / --input    — One or more DNA/Protein FASTA files; .gz supported. Deflines >ID|class (class optional).
-o / --output   — Output CSV path.
-t / --type {dna,protein} — Choose alphabet.
-k — k-mer length (e.g., 3).
--stride        — Step between k-mers (default 1).
--ambig         {skip,keep} — Ambiguity handling for tokens:
                skip: drop any k-mer containing ambiguous letters;
                keep: keep tokens verbatim.
--pretrained    PATH — Load a pretrained Doc2Vec .model and infer vectors for your sequences.
                Training hyperparameters (used when not --pretrained):
--vector-size (embedding dim)
--window (context window for PV-DM)
--epochs (training passes)
--min-count (token frequency cutoff)
--dm {1,0} (1=PV-DM, 0=PV-DBOW)
--dm-mean {1,0} (PV-DM averaging of context; default 1)
--dm-concat {1,0} (PV-DM concatenation of context; default 0)
--hs {1,0} / --negative / --sample (training objective & subsampling)
--seed, --workers
--save-model PATH — Save the trained Doc2Vec model (optional).
Inference/options:
--infer-steps, --infer-alpha — Params for infer_vector (used with pretrained or --force-infer).
--force-infer — Even after training, infer doc vectors instead of reading model.dv[tag].
--l2norm — L2-normalize the final sequence embedding.
--reverse-complement-merge — DNA only: average forward & reverse-complement doc vectors (strand-agnostic).
Misc:
--uppercase — Uppercase sequences; DNA U→T.
--minlen — Skip sequences shorter than this (after cleaning).
--add-length — Append Seq_length to CSV.
--progress — Progress bar (needs tqdm).

Example:

# 1) Train Doc2Vec PV-DM on DNA 6-mers, infer after training (force), RC-merge, L2 norm
python3 doc2vec_embeddings.py -i dna.fa -o dna_d2v_pvdm.csv -t dna -k 6 \
  --dm 1 --dm-mean 1 --window 5 --vector-size 200 --epochs 15 \
  --force-infer --infer-steps 80 --infer-alpha 0.03 \
  --reverse-complement-merge --l2norm --progress

# 2) Train Doc2Vec PV-DBOW on protein 3-mers, use trained docvecs directly, save model
python3 doc2vec_embeddings.py -i prots.fa.gz -o prot_d2v_dbow.csv -t protein -k 3 \
  --dm 0 --vector-size 128 --epochs 20 --min-count 2 --workers 8 \
  --save-model prot_doc2vec_dbow.model --progress

# 3) Use a pretrained Doc2Vec model, stride=2, skip ambiguous tokens
python3 doc2vec_embeddings.py -i cds.fasta -o cds_d2v_pretrained.csv -t dna -k 5 \
  --stride 2 --pretrained ref_doc2vec.model --ambig skip \
  --infer-steps 100 --infer-alpha 0.02 --progress


"""
