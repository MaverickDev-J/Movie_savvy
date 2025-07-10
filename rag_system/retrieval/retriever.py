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

# Define paths and settings from config
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['index_file'])
INDEX_METADATA_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['metadata_file'])
CHUNKS_FILE = os.path.join(config['data']['processed_dir'], config['data']['preprocessing']['output_file'])
MODEL_NAME = config['retrieval']['embedding_model']
TOP_K = config['retrieval']['top_k']
SIMILARITY_THRESHOLD = config['retrieval']['similarity_threshold']

async def load_index_and_metadata():
    """Load FAISS index and metadata asynchronously."""
    try:
        logger.info("Loading FAISS index and metadata...")
        index = await asyncio.to_thread(faiss.read_index, str(INDEX_FILE))
        with open(INDEX_METADATA_FILE, 'r') as f:
            index_metadata = json.load(f)
        with open(CHUNKS_FILE, 'r') as f:
            chunks = {json.loads(line)['id']: json.loads(line) for line in f}
        logger.info("Index and metadata loaded successfully")
        return index, index_metadata, chunks
    except Exception as e:
        logger.error(f"Failed to load index or metadata: {e}")
        sys.exit(1)

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
    """Retrieve top-k chunks asynchronously."""
    try:
        # Reload config to ensure the latest MODEL_NAME is used
        config = load_config()
        global MODEL_NAME, SIMILARITY_THRESHOLD
        MODEL_NAME = config['retrieval']['embedding_model']
        SIMILARITY_THRESHOLD = config['retrieval']['similarity_threshold']
        EMBEDDING_DIMENSION = config['retrieval']['embedding_dimension']
        
        # Preprocess query
        processed_query = preprocess_query(query)
        logger.info(f"Original query: {query}")
        
        # Load model and encode query with proper prefix
        logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        # Add "query: " prefix for E5 models
        prefix = "query: " if "e5" in MODEL_NAME.lower() else ""
        query_emb = await asyncio.to_thread(
            model.encode, [f"{prefix}{processed_query}"], convert_to_numpy=True, normalize_embeddings=True
        )
        query_emb = query_emb.astype(np.float32)
        
        # Load index and search
        index, index_metadata, chunks = await load_index_and_metadata()
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
        sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks for a query")
    parser.add_argument('--query', required=True, help="Query string")
    parser.add_argument('--top_k', type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()
    
    try:
        results = await retrieve(args.query, args.top_k)
        for i, res in enumerate(results, 1):
            logger.info(f"Result {i}: {res['source']} (Similarity: {res['similarity']:.4f}, Distance: {res['distance']:.4f})")
            logger.info(f"Text: {res['text'][:100]}...")
            logger.info(f"Metadata: {res['metadata']}\n")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
