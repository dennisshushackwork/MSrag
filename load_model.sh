# This script loads the embedding model from huggingface.
# Make sure you do have git lfs installed. https://git-lfs.com/
# Make also sure you have a huggingface account and added the access token via huggingface-cli login

# 1. Create a directory to store your models (if you don't have one)
mkdir -p ./models

# 2. Navigate into the directory
cd ./models

# Download the embedding model and the tokeniser (gemma)
git clone https://huggingface.co/Alibaba-NLP/gte-multilingual-base

# Make sure git-lfs pulls the actual model files
cd gte-multilingual-base
git lfs pull

# Downloading the gemma model (tokeniser needed only)
cd ..
git clone https://huggingface.co/google/gemma-3-1b-it
cd gemma-3-1b-it
git lfs pull

# Downloading qwen embedder_mlr_test:
cd ..
git clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
git lfs pull

# Download and save re-ranker:
cd ..
git clone https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
git lfs pull
