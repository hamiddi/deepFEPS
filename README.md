# deepFEPS — Deep Feature Extraction for Biological Sequences  🧬✨

**DeepFEPS** is a high-performance bioinformatics platform for extracting advanced sequence-based features from DNA, RNA, and protein data. It integrates modern machine learning and deep learning techniques to transform raw biological sequences into rich numerical representations suitable for classification, clustering, and predictive modeling.

Each feature extractor below offers an advanced way of representing biological sequences — from sequence embedding models such as Word2Vec, FastText, and Doc2Vec, to Transformer-based architectures, Autoencoder-derived features, and Graph-based embeddings. These deep learning and graph representation techniques can capture complex sequence patterns and relationships beyond simple k-mer counts, enabling more powerful analysis for functional annotation, motif discovery, and predictive modeling.

Simply select the method that best fits your research goals, upload your sequences, configure the parameters, and download your processed features.


<p align="left">
  <a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  </a>
  <a href="https://huggingface.co/docs/transformers/index">
    <img alt="Transformers" src="https://img.shields.io/badge/Transformers-🤗-FFD21E?labelColor=222222">
  </a>
  <a href="https://radimrehurek.com/gensim/">
    <img alt="Gensim" src="https://img.shields.io/badge/Gensim-Text%20Embeddings-66CC99">
  </a>
  <a href="https://networkx.org/">
    <img alt="NetworkX" src="https://img.shields.io/badge/NetworkX-Graphs-1177AA">
  </a>
  <a href="https://github.com/benedekrozemberczki/karateclub">
    <img alt="KarateClub" src="https://img.shields.io/badge/KarateClub-Graph%20Embeddings-FF6F61">
  </a>
  <a href="./LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
  </a>
</p>

**deepFEPS** is a command‑line toolkit for extracting **rich numerical features** from DNA/RNA/protein sequences in FASTA format. It bundles five complementary extractors:

- 🧠 **Autoencoder features** — learned representations from k‑mer bag‑of‑words or fixed one‑hot encodings (PyTorch).
- 📄 **Doc2Vec embeddings** — PV‑DM / PV‑DBOW document embeddings over k‑mers (Gensim).
- 🕸️ **Graph embeddings** — DeepWalk / Node2Vec / Graph2Vec on k‑mer graphs (NetworkX + Gensim + KarateClub).
- 🤖 **Transformer embeddings** — ProtBERT / ESM2 / DNABERT / custom HF models (Hugging Face Transformers).
- 🔤 **Word2Vec / FastText** — train embeddings on k‑mers, pool to fixed‑size vectors (Gensim).

Outputs are **CSV** files (one row per sequence).

---

## 1) 🛠️ Software & Hardware Requirements

### Software
- 🐍 **Python 3.10+** (tested with 3.10/3.11)
- 💻 OS: Linux or macOS recommended; Windows works but may require minor shell changes.
- 📦 Packages (managed via `requirements.txt`):
  - gensim
  - huggingface-hub
  - karateclub
  - networkx
  - numpy
  - torch
  - tqdm
  - transformers

### Hardware
- 🧮 **CPU**: All extractors work on CPU. Doc2Vec/Word2Vec/Graph are CPU‑friendly.
- ⚡ **GPU** (optional, recommended for Transformers/Autoencoder):
  - 🧩 NVIDIA CUDA 11+ compatible GPU.
  - 💾 **≥4 GB VRAM** works for small transformer models (e.g., ESM2‑t6). **8–16 GB+** recommended for larger models and batched inference.
  - 🐢 If you don’t have a GPU, everything runs on CPU; it will just be slower for Transformers.

> 💡 **Tip**: for servers without write access to default cache dirs, set a writable HF cache, e.g.:
> ```bash
> export HF_HOME=/path/to/hf-cache
> export HF_HUB_DISABLE_PROGRESS_BARS=1
> ```

---

## 2) ⚡ Quickstart (Anaconda)

```bash
conda env create -f environment.yml
conda activate deepfeps
# GPU alternative:
# conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python scripts/transformer_embeddings.py -i examples/sample.fasta -o features.csv -t protein --method esm2 --pool mean
```

---

## 3) 📥 Input Format

All extractors take **FASTA** files (`-i` accepts one or more files). DNA/RNA/protein is controlled by `-t {dna,protein}`. Compressed FASTA (`.gz`) is supported where noted by the scripts.

Example (`examples/sample.fasta`):
```fasta
>seq1
ACGTACGTACGTACGTACGTACGTACGTACGT
>seq2
ACGTACGTACGTACGTACGTACGTACGTACGT
```

---

## 4) 🧪 Feature Extractors

### A) 🧠 Autoencoder features  — `scripts/autoencoder_features.py`
Learn a compressed representation from either **k‑mer bag‑of‑words** or a **fixed one‑hot** encoding.

**Basic usage**
```bash
python scripts/autoencoder_features.py -i my.fasta -o autoenc.csv -t dna   --encoding kmer-bow -k 3 --normalize l2 --epochs 10 --batch-size 64 --device auto
```

**Common options**
- `--encoding` `kmer-bow|onehot-fixed` — representation
- `-k` — k‑mer size (kmer-bow)
- `--normalize` `none|l1|l2`
- `--hidden-dims` (e.g., `512 128`) and `--latent-dim`
- `--act` `relu|gelu|tanh`, `--dropout`, `--batch-norm`
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`
- `--device` `auto|cpu|cuda`, misc: `--reverse-complement-merge`, `--uppercase`, `--minlen`, `--l2norm`, `--add-length`, `--progress`

---

### B) 📄 Doc2Vec embeddings — `scripts/doc2vec_embeddings.py`
Treat each sequence as a document of k‑mers; learn **PV‑DM / PV‑DBOW** embeddings.

**Basic usage**
```bash
python scripts/doc2vec_embeddings.py -i my.fasta -o d2v.csv -t dna -k 3 --dm 1 --vector-size 100 --window 5 --epochs 20
```

**Common options**
- `--dm {1,0}`, `--dm-mean {1,0}`, `--dm-concat {1,0}`
- `--vector-size`, `--window`, `--epochs`, `--min-count`
- `--negative`, `--hs {1,0}`, `--sample`, `--seed`, `--workers`
- `--pretrained`, `--save-model`, `--infer-steps`, `--infer-alpha`, `--force-infer`
- Misc: `--reverse-complement-merge`, `--uppercase`, `--minlen`, `--l2norm`, `--add-length`, `--progress`

---

### C) 🕸️ Graph embeddings — `scripts/graph_embeddings.py`
Build a **k‑mer graph** (nodes=k‑mers, edges by co‑occurrence / overlaps), then embed via **DeepWalk / Node2Vec / Graph2Vec**.

**Basic usage**
```bash
python scripts/graph_embeddings.py -i my.fasta -o graph.csv -t dna -k 3 --method deepwalk --vector-size 128 --window 5 --epochs 5
```

**Common options**
- `--method` `deepwalk|node2vec|graph2vec`
- Graph2Vec/skip‑gram params: `--vector-size`, `--window`, `--epochs`
- Misc: `--reverse-complement-merge`, `--uppercase`, `--minlen`, `--add-length`, `--progress`
> Requires: `networkx`, `gensim`, `karateclub`.

---

### D) 🤖 Transformer embeddings — `scripts/transformer_embeddings.py`
Extract contextual embeddings using **Hugging Face** models.

**Basic usage (protein)**
```bash
python scripts/transformer_embeddings.py -i my.fasta -o esm2.csv -t protein   --method esm2 --pool mean --batch-size 4 --device cpu
```

**Basic usage (DNA/DNABERT)**
```bash
python scripts/transformer_embeddings.py -i my.fasta -o dnabert.csv -t dna   --method dnabert --k-mer 6 --pool mean --batch-size 2 --device cpu
```

**Common options**
- `--method` `protbert|esm2|dnabert|auto`
- `--model` override HF model id/path (when `--method auto`)
- `--k-mer` (DNABERT)
- `--pool` `cls|mean|mean+max`, `--batch-size`, `--device`

---

### E) 🔤 Word2Vec / FastText — `scripts/w2v_ft_embeddings.py`
Train **Word2Vec** or **FastText** on k‑mers and pool per sequence.

**Basic usage**
```bash
python scripts/w2v_ft_embeddings.py -i my.fasta -o w2v.csv -t dna   -k 3 --algo word2vec --vector-size 100 --window 5 --epochs 10 --pool mean
```

**Common options**
- `--algo` `word2vec|fasttext`, `--vector-size`, `--window`, `--epochs`
- `--pool` `mean|sum|max|mean+max`
- Misc: `--reverse-complement-merge`, `--uppercase`, `--minlen`, `--l2norm`, `--add-length`, `--progress`

---

## 5) 📤 Output
CSV with `id` (FASTA header), optional `class`, then feature columns.

## 6) 🔁 Reproducibility & Tips
- 🎲 Set `--seed` where available.
- 🧱 For long sequences with BERT‑style models (positional limit ~512), pre‑truncate or run windows externally.
- 🗂️ Configure HF cache on servers (see above tip).
---

## 🙌 Citation

```bibtex
@software{deepfeps_2025,
  title        = {deepFEPS: Deep Learning-Oriented Feature Extraction for Biological Sequences},
  year         = {2025},
  url          = {https://github.com/<your-org>/deepFEPS}
}
```

## 📄 License

MIT (see `LICENSE`).
