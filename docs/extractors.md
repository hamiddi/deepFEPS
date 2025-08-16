
# Extractor details

This page expands on each extractor with practical guidance and defaults.

## Autoencoder
- **Tokenization**: `kmer-bow` (sparse hashed or explicit) or `onehot-fixed` (pad/truncate).
- **Normalization**: Start with `--normalize l2`.
- **Training tips**: Increase `--epochs`, reduce `--batch-size` for large datasets.

## Transformer
- **ESM2/ProtBERT** are protein models; **DNABERT** is for DNA (requires `--k-mer`).
- If you see the **pooler weights warning**, it's benign for feature extraction.
- Keep sequences within positional limits or window them before running.

## Doc2Vec
- `--dm 1` (PV-DM) uses context words to predict target k-mers; good default.
- `--dm 0` (PV-DBOW) is faster but may require more epochs.

## Word2Vec / FastText
- `word2vec` is a solid default; `fasttext` helps with rare k-mers via subword units.
- Pooling: mean pooling is robust; mean+max can capture salient motifs.

## Graph
- Build a graph where nodes are k-mers; edges represent adjacency or co-occurrence.
- `deepwalk` is simple and effective; `graph2vec` treats graphs as documents.
