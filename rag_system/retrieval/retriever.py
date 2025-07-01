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

async def retrieve(query, top_k=TOP_K):
    """Retrieve top-k chunks asynchronously using cached index."""
    try:
        # Reload config to ensure the latest settings
        config = load_config()
        global SIMILARITY_THRESHOLD
        SIMILARITY_THRESHOLD = config['retrieval']['similarity_threshold']
        MODEL_NAME = config['retrieval']['embedding_model']
        
        # Preprocess query
        processed_query = preprocess_query(query)
        logger.info(f"Original query: {query}")
        
        # Get cached model (loads once, then reuses)
        model = await index_manager.get_model()
        
        # Add "query: " prefix for E5 models
        prefix = "query: " if "e5" in MODEL_NAME.lower() else ""
        query_emb = await asyncio.to_thread(
            model.encode, [f"{prefix}{processed_query}"], convert_to_numpy=True, normalize_embeddings=True
        )
        query_emb = query_emb.astype(np.float32)
        
        # Get cached index and data (loads once, then reuses)
        index, index_metadata, chunks = await index_manager.get_index_and_data()
        
        logger.info(f"Searching index for top {top_k} chunks...")
        distances, indices = await asyncio.to_thread(index.search, query_emb, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # Ensure valid index
                meta = index_metadata[idx]
                chunk = chunks.get(meta['chunk_id'])
                if chunk is None:
                    logger.warning(f"Chunk with ID {meta['chunk_id']} not found in chunks file")
                    continue
                similarity = float(1 - dist)  # Convert distance to similarity
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append({
                        'chunk_id': meta['chunk_id'],
                        'source': meta['source'],
                        'text': chunk['text'],
                        'metadata': chunk['metadata'],
                        'distance': float(dist),
                        'similarity': similarity
                    })
        
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return e  # Return exception instead of sys.exit for better error handling

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