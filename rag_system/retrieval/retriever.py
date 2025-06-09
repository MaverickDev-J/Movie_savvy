# import json
# import numpy as np
# import faiss
# import logging
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# import argparse

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# BASE_DIR = Path(__file__).resolve().parent.parent
# INDEX_FILE = BASE_DIR / "retrieval" / "index" / "index.bin"
# INDEX_METADATA_FILE = BASE_DIR / "retrieval" / "index" / "index_metadata.json"
# CHUNKS_FILE = BASE_DIR / "data" / "processed" / "processed_chunks.jsonl"

# def load_index_and_metadata():
#     index = faiss.read_index(str(INDEX_FILE))
#     with open(INDEX_METADATA_FILE, 'r') as f:
#         index_metadata = json.load(f)
#     with open(CHUNKS_FILE, 'r') as f:
#         chunks = {json.loads(line)['id']: json.loads(line) for line in f}
#     return index, index_metadata, chunks

# def retrieve(query, top_k=5):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
#     index, index_metadata, chunks = load_index_and_metadata()
#     distances, indices = index.search(query_emb, top_k)
#     results = []
#     for idx, dist in zip(indices[0], distances[0]):
#         meta = index_metadata[idx]
#         chunk = chunks[meta['chunk_id']]
#         results.append({
#             'chunk_id': meta['chunk_id'],
#             'source': meta['source'],
#             'text': chunk['text'],
#             'metadata': chunk['metadata'],
#             'distance': float(dist)
#         })
#     return results

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--query', required=True)
#     parser.add_argument('--top_k', type=int, default=5)
#     args = parser.parse_args()
#     results = retrieve(args.query, args.top_k)
#     for i, res in enumerate(results, 1):
#         logger.info(f"Result {i}: {res['source']} (Distance: {res['distance']:.4f})")
#         logger.info(f"Text: {res['text'][:100]}...")
#         logger.info(f"Metadata: {res['metadata']}\n")

# if __name__ == "__main__":
#     main()







# import json
# import numpy as np
# import faiss
# import logging
# import asyncio
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# import argparse
# import re

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# BASE_DIR = Path(__file__).resolve().parent.parent
# INDEX_FILE = BASE_DIR / "retrieval" / "index" / "index.bin"
# INDEX_METADATA_FILE = BASE_DIR / "retrieval" / "index" / "index_metadata.json"
# CHUNKS_FILE = BASE_DIR / "data" / "processed" / "processed_chunks.jsonl"

# async def load_index_and_metadata():
#     """Load FAISS index and metadata asynchronously."""
#     logger.info("Loading FAISS index and metadata...")
#     index = await asyncio.to_thread(faiss.read_index, str(INDEX_FILE))
#     with open(INDEX_METADATA_FILE, 'r') as f:
#         index_metadata = json.load(f)
#     with open(CHUNKS_FILE, 'r') as f:
#         chunks = {json.loads(line)['id']: json.loads(line) for line in f}
#     logger.info("Index and metadata loaded successfully")
#     return index, index_metadata, chunks

# def preprocess_query(query):
#     """Preprocess query to focus on key terms for diverse movie data."""
#     filler_words = {
#         'best', 'top', 'good', 'great', 'movie', 'film', 'kmovie', 'kdrama', 'anime', 'manga',
#         'bollywood', 'hollywood', 'on', 'about', 'what', 'is', 'are', 'the', 'a', 'an',
#         'in', 'of', 'for', 'to', 'and', 'with', 'how', 'why', 'which'
#     }
#     query = query.lower()
#     # Split into words, preserving phrases like "train to busan"
#     words = re.findall(r'\b[\w\s-]+\b', query)  # Match words and phrases
#     key_terms = [w.strip() for w in words if w.strip() not in filler_words and len(w.strip()) > 3]
#     processed = ' '.join(key_terms) if key_terms else query
#     logger.info(f"Preprocessed query: {processed}")
#     return processed

# async def retrieve(query, top_k=5):
#     """Retrieve top-k chunks asynchronously using SentenceTransformer."""
#     try:
#         # Preprocess query
#         processed_query = preprocess_query(query)
#         logger.info(f"Original query: {query}")
        
#         # Load model and encode query
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         query_emb = await asyncio.to_thread(
#             model.encode, [processed_query], convert_to_numpy=True, normalize_embeddings=True
#         )
#         query_emb = query_emb.astype(np.float32)
        
#         # Load index and search
#         index, index_metadata, chunks = await load_index_and_metadata()
#         logger.info(f"Searching index for top {top_k} chunks...")
#         distances, indices = await asyncio.to_thread(index.search, query_emb, top_k)
        
#         results = []
#         for idx, dist in zip(indices[0], distances[0]):
#             if idx >= 0:  # Ensure valid index
#                 meta = index_metadata[idx]
#                 chunk = chunks[meta['chunk_id']]
#                 similarity = float(1 - dist)  # Convert distance to similarity
#                 results.append({
#                     'chunk_id': meta['chunk_id'],
#                     'source': meta['source'],
#                     'text': chunk['text'],
#                     'metadata': chunk['metadata'],
#                     'distance': float(dist),
#                     'similarity': similarity
#                 })
        
#         logger.info(f"Retrieved {len(results)} chunks for query")
#         return results
#     except Exception as e:
#         logger.error(f"Retrieval failed: {e}")
#         raise

# async def main():
#     parser = argparse.ArgumentParser(description="Retrieve relevant chunks for a query")
#     parser.add_argument('--query', required=True, help="Query string")
#     parser.add_argument('--top_k', type=int, default=5, help="Number of chunks to retrieve")
#     args = parser.parse_args()
    
#     try:
#         results = await retrieve(args.query, args.top_k)
#         for i, res in enumerate(results, 1):
#             logger.info(f"Result {i}: {res['source']} (Similarity: {res['similarity']:.4f}, Distance: {res['distance']:.4f})")
#             logger.info(f"Text: {res['text'][:100]}...")
#             logger.info(f"Metadata: {res['metadata']}\n")
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())





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

# Load configuration from rag_config.yaml
try:
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "rag_config.yaml")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load rag_config.yaml: {e}")
    sys.exit(1)

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
        # Preprocess query
        processed_query = preprocess_query(query)
        logger.info(f"Original query: {query}")
        
        # Load model and encode query with proper prefix
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
                chunk = chunks[meta['chunk_id']]
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