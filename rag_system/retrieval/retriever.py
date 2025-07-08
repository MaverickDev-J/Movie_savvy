import os
import json
import numpy as np
import faiss
import logging
import asyncio
import sys
import yaml
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse
import re
from typing import List, Dict
from rag_system.embeddings.chunk_embeddings import ChunkEmbeddingManager
from .index_manager import index_manager  # Import the singleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from rag_config.yaml."""
    try:
        CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "rag_config.yaml")
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load rag_config.yaml: {e}")
        sys.exit(1)

# Load configuration
config = load_config()

# Define settings from config
TOP_K = config['retrieval']['top_k']
SIMILARITY_THRESHOLD = config['retrieval']['similarity_threshold']

# REMOVE THIS FUNCTION - replaced by index_manager
# async def load_index_and_metadata():
#     """This function is now handled by IndexManager"""
#     pass

def preprocess_query(query):
    """Preprocess query to focus on key terms for diverse movie data."""
    filler_words = {
        'best', 'top', 'good', 'great', 'movie', 'film', 'kmovie', 'kdrama', 'anime', 'manga',
        'bollywood', 'hollywood', 'on', 'about', 'what', 'is', 'are', 'the', 'a', 'an',
        'in', 'of', 'for', 'to', 'and', 'with', 'how', 'why', 'which'
    }
    query = query.lower()
    # Split into words, preserving phrases like "train to busan"
    words = re.findall(r'\b[\w\s-]+\b', query)  # Match words and phrases
    key_terms = [w.strip() for w in words if w.strip() not in filler_words and len(w.strip()) > 3]
    processed = ' '.join(key_terms) if key_terms else query
    logger.info(f"Preprocessed query: {processed}")
    return processed

async def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    try:
        embedding_manager = ChunkEmbeddingManager()
        embedding_manager.load_index()
        chunks = embedding_manager.search_index(query, top_k)
        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return chunks
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

# Rest of your code remains the same...
async def main():
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks for a query")
    parser.add_argument('--query', required=True, help="Query string")
    parser.add_argument('--top_k', type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()
    
    try:
        results = await retrieve(args.query, args.top_k)
        if isinstance(results, Exception):
            logger.error(f"Retrieval failed: {results}")
            return
            
        for i, res in enumerate(results, 1):
            logger.info(f"Result {i}: {res['source']} (Similarity: {res['similarity']:.4f}, Distance: {res['distance']:.4f})")
            logger.info(f"Text: {res['text'][:100]}...")
            logger.info(f"Metadata: {res['metadata']}\n")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())