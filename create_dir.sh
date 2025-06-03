#!/bin/bash

# Create main project directory
mkdir -p rag_system

# Create subdirectories
mkdir -p rag_system/data/raw
mkdir -p rag_system/data/processed/embeddings
mkdir -p rag_system/models/mistral-7b-finetuned
mkdir -p rag_system/retrieval/index
mkdir -p rag_system/generation/prompt_templates
mkdir -p rag_system/config
mkdir -p rag_system/scripts
mkdir -p rag_system/output/logs
mkdir -p rag_system/output/results

# Create empty files in data/raw
touch rag_system/data/raw/anime.jsonl
touch rag_system/data/raw/hollywood.jsonl
touch rag_system/data/raw/bollywood.jsonl
touch rag_system/data/raw/manga.jsonl
touch rag_system/data/raw/kdrama.jsonl
touch rag_system/data/raw/kmovie.jsonl

# Create empty metadata file
touch rag_system/data/processed/metadata.json

# Create empty scripts
touch rag_system/scripts/preprocess_data.py
touch rag_system/scripts/build_index.py
touch rag_system/scripts/run_rag.py
touch rag_system/retrieval/embedder.py
touch rag_system/generation/generator.py

# Create empty config files
touch rag_system/config/rag_config.yaml
touch rag_system/config/model_config.yaml

# Create empty prompt template file
touch rag_system/generation/prompt_templates/rag_prompt.txt

# Copy fine-tuned model files from existing directory
cp -r out/qlora-mistral-7b-custom-data/final/* rag_system/models/mistral-7b-finetuned/

# Copy base model files
cp -r checkpoints/mistralai/Mistral-7B-v0.1/* rag_system/models/mistral-7b-finetuned/

# Set permissions
chmod -R 755 rag_system

echo "RAG system directory structure created successfully."