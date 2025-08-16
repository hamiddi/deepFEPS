
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY scripts/ /app/scripts/

# Default command shows help for each extractor
CMD python /app/scripts/autoencoder_features.py -h || true &&     python /app/scripts/doc2vec_embeddings.py -h || true &&     python /app/scripts/graph_embeddings.py -h || true &&     python /app/scripts/w2v_ft_embeddings.py -h || true &&     python /app/scripts/transformer_embeddings.py -h || true
