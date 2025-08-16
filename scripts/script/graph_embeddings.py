#!/usr/bin/env python3
"""
graph_embeddings.py
-------------------
Graph embeddings from DNA/Protein FASTA → CSV.

Per sequence:
  - Build a k-mer co-occurrence graph (nodes = k-mers; edges between successive k-mers).
  - Embed the graph using one of:
      * deepwalk  : uniform random walks + Word2Vec
      * node2vec  : biased random walks + Word2Vec (p, q)
      * graph2vec : whole-graph embedding (karateclub)
  - Pool node embeddings to a fixed-length vector (mean/sum/max/mean+max).
  - (DNA) Optional reverse-complement merge at the sequence-embedding level.

Output:
  CSV with columns: ID, EMB_*, ..., class   (optionally Seq_length appended before class)

Notes:
  - Methods here do not benefit meaningfully from GPU; --device is accepted and safe,
    but computation runs on CPU. No Triton is used.
"""

import argparse
import csv
import gzip
import os
import sys
import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

try:
    from gensim.models import Word2Vec
except Exception:
    print("[error] This script needs gensim. Try: pip install gensim tqdm networkx numpy", file=sys.stderr)
    sys.exit(1)

# ----------------------------
# FASTA utils
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
# k-mer graph construction
# ----------------------------

DNA_VALID = set("ACGT")
AA_VALID  = set("ACDEFGHIKLMNPQRSTVWY")
AA_AMBIG  = set(["B","Z","X"])  # only used if ambig=keep

def tokenize_kmers(seq: str, k: int, stride: int, seq_type: str, ambig: str) -> List[str]:
    n = len(seq)
    out = []
    if ambig == "skip":
        if seq_type == "dna":
            for i in range(0, n - k + 1, stride):
                tok = seq[i:i+k]
                if set(tok) <= DNA_VALID:
                    out.append(tok)
        else:
            for i in range(0, n - k + 1, stride):
                tok = seq[i:i+k]
                if set(tok).issubset(AA_VALID):
                    out.append(tok)
    else:  # keep
        for i in range(0, n - k + 1, stride):
            out.append(seq[i:i+k])
    return out

def build_kmer_graph(tokens: List[str], directed: bool=False) -> nx.Graph:
    """
    Build a (un)directed k-mer co-occurrence graph where edges connect consecutive tokens.
    Edge weight = count of transitions.
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    if not tokens:
        return G

    for t in tokens:
        if not G.has_node(t):
            G.add_node(t)

    for a, b in zip(tokens, tokens[1:]):
        if a == "" or b == "":
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1.0
        else:
            G.add_edge(a, b, weight=1.0)

    return G

# ----------------------------
# Random-walk generators
# ----------------------------

def _choose_neighbor_uniform(G: nx.Graph, node: str, rng: random.Random):
    nbrs = list(G.neighbors(node))
    if not nbrs:
        return None
    return rng.choice(nbrs)

def _node2vec_next(G: nx.Graph, prev: str, cur: str, p: float, q: float, rng: random.Random):
    """
    Node2Vec transition: from prev -> cur, choose next with biases:
      return to prev: 1/p
      neighbors of prev: 1
      others: 1/q
    """
    nbrs = list(G.neighbors(cur))
    if not nbrs:
        return None
    weights = []
    for x in nbrs:
        if x == prev:
            w = 1.0 / p
        elif G.has_edge(x, prev) or G.has_edge(prev, x):
            w = 1.0
        else:
            w = 1.0 / q
        # incorporate edge weight if present
        ew = G[cur][x].get("weight", 1.0)
        weights.append(w * ew)
    s = sum(weights)
    if s <= 0:
        return rng.choice(nbrs)
    probs = [w / s for w in weights]
    r = rng.random()
    acc = 0.0
    for x, pr in zip(nbrs, probs):
        acc += pr
        if r <= acc:
            return x
    return nbrs[-1]

def generate_walks(G: nx.Graph, num_walks: int, walk_length: int,
                   method: str, p: float=1.0, q: float=1.0, seed: int=1) -> List[List[str]]:
    """
    Create a list of node sequences (sentences) for Word2Vec.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []
    if not nodes:
        return walks

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            if method == "deepwalk":
                while len(walk) < walk_length:
                    nxt = _choose_neighbor_uniform(G, walk[-1], rng)
                    if nxt is None:
                        break
                    walk.append(nxt)
            else:  # node2vec
                if walk_length == 1:
                    walks.append(walk)
                    continue
                prev = start
                cur = _choose_neighbor_uniform(G, start, rng)
                if cur is None:
                    walks.append(walk)
                    continue
                walk.append(cur)
                while len(walk) < walk_length:
                    nxt = _node2vec_next(G, prev, cur, p, q, rng)
                    if nxt is None:
                        break
                    prev, cur = cur, nxt
                    walk.append(nxt)
            walks.append(walk)
    return walks

# ----------------------------
# Node embedding → sequence embedding
# ----------------------------

def pool_vectors(vecs: np.ndarray, mode: str="mean") -> np.ndarray:
    if vecs.size == 0:
        return vecs
    if mode == "mean":
        return vecs.mean(axis=0)
    if mode == "sum":
        return vecs.sum(axis=0)
    if mode == "max":
        return vecs.max(axis=0)
    if mode == "mean+max":
        return np.concatenate([vecs.mean(axis=0), vecs.max(axis=0)], axis=0)
    return vecs.mean(axis=0)

def graph_embed_deepwalk_node2vec(G: nx.Graph, method: str,
                                  vector_size: int, window: int, epochs: int,
                                  sg: int, num_walks: int, walk_length: int,
                                  p: float, q: float, seed: int,
                                  pool: str="mean") -> np.ndarray:
    if len(G.nodes) == 0:
        return np.zeros(2*vector_size if pool=="mean+max" else vector_size, dtype=float)

    walks = generate_walks(G, num_walks=num_walks, walk_length=walk_length,
                           method=method, p=p, q=q, seed=seed)
    if len(walks) == 0:
        return np.zeros(2*vector_size if pool=="mean+max" else vector_size, dtype=float)

    model = Word2Vec(
        sentences=walks,
        vector_size=vector_size,
        window=window,
        min_count=0,
        sg=sg,
        workers=1,     # deterministic across environments; graphs are small per seq
        epochs=epochs,
        seed=seed
    )
    # Collect node vectors in this graph
    vec_list = []
    for n in G.nodes():
        if n in model.wv:
            vec_list.append(model.wv.get_vector(n))
    if not vec_list:
        return np.zeros(2*vector_size if pool=="mean+max" else vector_size, dtype=float)
    V = np.stack(vec_list, axis=0)
    return pool_vectors(V, pool)
"""
def graph_embed_graph2vec(graphs: List[nx.Graph], dim: int, wl_iterations: int, seed: int):
    try:
        from karateclub import Graph2Vec
    except Exception:
        print("[error] Method 'graph2vec' requires karateclub. Try: pip install karateclub", file=sys.stderr)
        sys.exit(1)
    model = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations, seed=seed, workers=1)  # workers=1 for portability
    model.fit(graphs)
    return model.get_embedding()  # np.ndarray [N, dim]
"""
def _normalize_for_graph2vec(G: nx.Graph) -> nx.Graph:
    """
    Ensure Graph2Vec-friendly format:
      - Undirected graph
      - Non-empty (add a dummy node if needed)
      - Integer node labels 0..n-1
    """
    # Force undirected
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected(as_view=False)
    else:
        G = nx.Graph(G)

    # Ensure non-empty
    if G.number_of_nodes() == 0:
        G.add_node(0)

    # Relabel nodes to 0..n-1
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G

def graph_embed_graph2vec(graphs: List[nx.Graph], dim: int, wl_iterations: int, seed: int):
    try:
        from karateclub import Graph2Vec
    except Exception:
        print("[error] Method 'graph2vec' requires karateclub. Try: pip install karateclub", file=sys.stderr)
        sys.exit(1)

    # Normalize every graph for karateclub
    graphs_norm = [_normalize_for_graph2vec(G) for G in graphs]

    model = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations, seed=seed, workers=1)
    model.fit(graphs_norm)
    return model.get_embedding()  # np.ndarray [N, dim]

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

# ----------------------------
# CLI & main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Graph embeddings (DeepWalk / Node2Vec / Graph2Vec) from DNA/Protein FASTA → CSV."
    )
    # IO
    p.add_argument("-i", "--input", nargs="+", required=True,
                   help="FASTA file(s) (.fa/.fasta[.gz]). Deflines: >ID|class (class optional).")
    p.add_argument("-o", "--output", required=True,
                   help="Output CSV file.")
    p.add_argument("-t", "--type", choices=["dna","protein"], required=True,
                   help="Sequence type.")
    # Graph building
    p.add_argument("-k", type=int, required=True,
                   help="k-mer length (e.g., 3).")
    p.add_argument("--stride", type=int, default=1,
                   help="Stride between consecutive k-mers (default: 1).")
    p.add_argument("--ambig", choices=["skip","keep"], default="skip",
                   help="Ambiguous letters: 'skip' drops tokens containing them; 'keep' keeps them verbatim.")
    p.add_argument("--undirected", action="store_true",
                   help="Build an undirected graph (default). If not set, a directed graph is built.")
    # Method
    p.add_argument("--method", choices=["deepwalk","node2vec","graph2vec"], default="deepwalk",
                   help="Graph embedding method.")
    # DeepWalk/Node2Vec params
    p.add_argument("--vector-size", type=int, default=128,
                   help="Embedding dimension for node embeddings (deepwalk/node2vec).")
    p.add_argument("--window", type=int, default=5,
                   help="Word2Vec context window for walks.")
    p.add_argument("--epochs", type=int, default=5,
                   help="Word2Vec training epochs.")
    p.add_argument("--sg", type=int, choices=[0,1], default=1,
                   help="Word2Vec architecture: 1=skip-gram, 0=CBOW (default=1).")
    p.add_argument("--num-walks", type=int, default=10,
                   help="Number of walks per node.")
    p.add_argument("--walk-length", type=int, default=40,
                   help="Number of nodes per walk.")
    p.add_argument("--p", type=float, default=1.0,
                   help="Return parameter p for node2vec (ignored for deepwalk).")
    p.add_argument("--q", type=float, default=1.0,
                   help="In-out parameter q for node2vec (ignored for deepwalk).")
    # Graph2Vec params
    p.add_argument("--g2v-dim", type=int, default=256,
                   help="Graph2Vec embedding dimension.")
    p.add_argument("--g2v-wl-iters", type=int, default=2,
                   help="Graph2Vec Weisfeiler-Lehman iterations.")
    # Pooling & post
    p.add_argument("--pool", choices=["mean","sum","max","mean+max"], default="mean",
                   help="Pool node vectors to sequence vector (deepwalk/node2vec). Ignored for graph2vec.")
    p.add_argument("--l2norm", action="store_true",
                   help="L2-normalize the final sequence embedding.")
    p.add_argument("--reverse-complement-merge", action="store_true",
                   help="DNA only: build RC graph too and average embeddings.")
    # Misc
    p.add_argument("--device", choices=["cpu","gpu"], default="cpu",
                   help="CPU or GPU flag (these methods use CPU; GPU choice is accepted and safe, no Triton used).")
    p.add_argument("--minlen", type=int, default=0,
                   help="Skip sequences shorter than this after cleaning.")
    p.add_argument("--add-length", action="store_true",
                   help="Append 'Seq_length' column.")
    p.add_argument("--save-vocab", type=str,
                   help="Save node vocabulary (k-mers) per sequence into a .txt (one file per sequence, named <ID>.txt) in the given folder.")
    p.add_argument("--progress", action="store_true",
                   help="Show a progress bar if tqdm is installed.")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed.")
    return p.parse_args()

def main():
    args = parse_args()

    # Read sequences
    records = list(read_fasta(args.input, args.type, minlen=args.minlen))
    if not records:
        print("No sequences passed filters.", file=sys.stderr); sys.exit(1)

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

    # Build graphs (forward and optional RC)
    graphs_fwd = []
    graphs_rc  = [] if (args.type=="dna" and args.reverse_complement_merge) else None
    node_vocab_per_id: Dict[str, List[str]] = {}

    loop_iter = zip(ids, seqs)
    if args.progress and HAS_TQDM:
        loop_iter = tqdm(loop_iter, total=len(ids), desc="Building graphs", unit="seq")

    for rid, s in loop_iter:
        toks = tokenize_kmers(s, args.k, args.stride, args.type, args.ambig)
        G = build_kmer_graph(toks, directed=not args.undirected)
        graphs_fwd.append(G)
        node_vocab_per_id[rid] = list(G.nodes())

        if graphs_rc is not None:
            s_rc = reverse_complement(s)
            toks_rc = tokenize_kmers(s_rc, args.k, args.stride, args.type, args.ambig)
            G_rc = build_kmer_graph(toks_rc, directed=not args.undirected)
            graphs_rc.append(G_rc)

    # Save per-sequence vocab (if requested)
    if args.save_vocab:
        os.makedirs(args.save_vocab, exist_ok=True)
        for rid, nodes in node_vocab_per_id.items():
            with open(os.path.join(args.save_vocab, f"{rid}.txt"), "w", encoding="utf-8") as fh:
                for n in nodes:
                    fh.write(n + "\n")

    # Embed graphs
    rng = np.random.default_rng(args.seed)
    seq_embeddings = []

    if args.method in ("deepwalk", "node2vec"):
        # Per-sequence independent training (keeps memory small and respects per-graph semantics)
        iterator = range(len(graphs_fwd))
        if args.progress and HAS_TQDM:
            iterator = tqdm(iterator, total=len(graphs_fwd), desc=f"Embedding ({args.method})", unit="seq")

        for i in iterator:
            emb_fwd = graph_embed_deepwalk_node2vec(
                graphs_fwd[i], method=args.method,
                vector_size=args.vector_size, window=args.window, epochs=args.epochs,
                sg=args.sg, num_walks=args.num_walks, walk_length=args.walk_length,
                p=args.p, q=args.q, seed=args.seed, pool=args.pool
            )

            if graphs_rc is not None:
                emb_rc = graph_embed_deepwalk_node2vec(
                    graphs_rc[i], method=args.method,
                    vector_size=args.vector_size, window=args.window, epochs=args.epochs,
                    sg=args.sg, num_walks=args.num_walks, walk_length=args.walk_length,
                    p=args.p, q=args.q, seed=args.seed, pool=args.pool
                )
                emb = 0.5*(emb_fwd + emb_rc)
            else:
                emb = emb_fwd

            if args.l2norm:
                emb = l2_normalize(emb)
            seq_embeddings.append(emb)

    else:  # graph2vec
        iterator = graphs_fwd
        if args.progress and HAS_TQDM:
            iterator = tqdm(graphs_fwd, total=len(graphs_fwd), desc="Embedding (graph2vec)", unit="seq")

        # graph2vec embeds the whole list at once; no RC-merge supported directly
        emb_fwd = graph_embed_graph2vec(list(iterator), dim=args.g2v_dim,
                                        wl_iterations=args.g2v_wl_iters, seed=args.seed)
        seq_embeddings = [e for e in emb_fwd]

        if graphs_rc is not None:
            # Also embed RC graphs and average
            emb_rc = graph_embed_graph2vec(graphs_rc, dim=args.g2v_dim,
                                           wl_iterations=args.g2v_wl_iters, seed=args.seed)
            seq_embeddings = [0.5*(a + b) for a, b in zip(seq_embeddings, emb_rc)]

        if args.l2norm:
            seq_embeddings = [l2_normalize(v) for v in seq_embeddings]

    seq_embeddings = np.asarray(seq_embeddings, dtype=float)
    dim = seq_embeddings.shape[1] if seq_embeddings.size else (2*args.vector_size if args.pool=="mean+max" else args.vector_size)

    # Column names
    if args.method in ("deepwalk","node2vec") and args.pool == "mean+max":
        half = dim // 2
        feature_names = [f"EMB_MEAN_{i+1}" for i in range(half)] + [f"EMB_MAX_{i+1}" for i in range(half)]
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
            row = [rid] + list(map(float, seq_embeddings[i].tolist()))
            if args.add_length:
                row.append(lengths[i])
            row.append(classes[i])
            w.writerow(row)

    print(f"Saved: {args.output}")
    print(f"Sequences: {len(ids)} | Feature columns (excluding ID/class): {len(feature_names)}")

if __name__ == "__main__":
    main()
"""
Options (quick reference)
------------------------
-i / --input — FASTA file(s); .gz supported; deflines >ID|class (class optional).
-o / --output — Output CSV.
-t / --type {dna,protein} — Choose alphabet.
-k — k-mer length (e.g., 3).
--stride — Step between k-mers (default 1).
--ambig {skip,keep} — Drop k-mers with ambiguous letters or keep them.
--undirected — If set, build an undirected graph (default behavior). If not set, graph is directed.
--method {deepwalk,node2vec,graph2vec} — Pick embedding method.

DeepWalk / Node2Vec
-------------------
--vector-size — Node embedding dimension (default 128).
--window — Word2Vec window over walk sequences (default 5).
--epochs — Word2Vec epochs (default 5).
--sg {1,0} — 1=skip-gram, 0=CBOW (default 1).
--num-walks — Walks per node (default 10).
--walk-length — Steps per walk (default 40).
--p, --q — Node2Vec bias parameters (ignored by DeepWalk).
Graph2Vec (optional dependency karateclub)
--g2v-dim — Graph2Vec embedding dimension (default 256).
--g2v-wl-iters — WL iterations (default 2).

Post / Misc
-----------
--pool {mean,sum,max,mean+max} — Pool node vectors to one sequence vector (ignored by graph2vec).
--l2norm — L2-normalize the final vector.
--reverse-complement-merge — DNA only: compute embedding on RC(seq) and average.
--device {cpu,gpu} — Accepted for interface consistency; computation runs on CPU either way. No Triton needed/used.
--minlen — Skip sequences shorter than this length.
--add-length — Append Seq_length.
--save-vocab DIR — Save per-sequence node lists (k-mer vocabulary) into DIR/<ID>.txt.
--progress — Show progress bars.
--seed — Random seed.

Examples
# 1) DeepWalk on DNA 4-mers, mean pooling, RC-merge, L2-normalize
python3 graph_embeddings.py -i dna.fa -o dna_dw_mean_rc.csv -t dna -k 4 \
  --method deepwalk --vector-size 128 --window 5 --epochs 5 \
  --num-walks 10 --walk-length 40 --pool mean \
  --reverse-complement-merge --l2norm --progress

# 2) Node2Vec on protein 3-mers, mean+max pooling
python3 graph_embeddings.py -i prots.fa.gz -o prot_n2v_meanmax.csv -t protein -k 3 \
  --method node2vec --vector-size 160 --window 5 --epochs 6 \
  --num-walks 10 --walk-length 30 --p 1.0 --q 2.0 \
  --pool mean+max --progress

# 3) Graph2Vec on DNA 5-mers, WL=3
python3 graph_embeddings.py -i promoters.fa -o dna_g2v.csv -t dna -k 5 \
  --method graph2vec --g2v-dim 256 --g2v-wl-iters 3 --progress

# 4) Save node vocabularies (k-mers) per sequence
python3 graph_embeddings.py -i cds.fa -o cds_n2v.csv -t dna -k 6 \
  --method node2vec --save-vocab node_vocab/ --progress



"""
