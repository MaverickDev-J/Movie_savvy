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



# base -----------------------------------------------------------------------

# import os 
# import json
# import numpy as np
# import faiss
# import logging
# import asyncio
# import sys
# import yaml
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# import argparse
# import re

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Load configuration from rag_config.yaml
# try:
#     CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "rag_config.yaml")
#     with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)
# except Exception as e:
#     logger.error(f"Failed to load rag_config.yaml: {e}")
#     sys.exit(1)

# # Define paths and settings from config
# BASE_DIR = Path(__file__).resolve().parent.parent
# INDEX_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['index_file'])
# INDEX_METADATA_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['metadata_file'])
# CHUNKS_FILE = os.path.join(config['data']['processed_dir'], config['data']['preprocessing']['output_file'])
# MODEL_NAME = config['retrieval']['embedding_model']
# TOP_K = config['retrieval']['top_k']
# SIMILARITY_THRESHOLD = config['retrieval']['similarity_threshold']

# async def load_index_and_metadata():
#     """Load FAISS index and metadata asynchronously."""
#     try:
#         logger.info("Loading FAISS index and metadata...")
#         index = await asyncio.to_thread(faiss.read_index, str(INDEX_FILE))
#         with open(INDEX_METADATA_FILE, 'r') as f:
#             index_metadata = json.load(f)
#         with open(CHUNKS_FILE, 'r') as f:
#             chunks = {json.loads(line)['id']: json.loads(line) for line in f}
#         logger.info("Index and metadata loaded successfully")
#         return index, index_metadata, chunks
#     except Exception as e:
#         logger.error(f"Failed to load index or metadata: {e}")
#         sys.exit(1)

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

# async def retrieve(query, top_k=TOP_K):
#     """Retrieve top-k chunks asynchronously."""
#     try:
#         # Preprocess query
#         processed_query = preprocess_query(query)
#         logger.info(f"Original query: {query}")
        
#         # Load model and encode query with proper prefix
#         model = SentenceTransformer(MODEL_NAME)
#         # Add "query: " prefix for E5 models
#         prefix = "query: " if "e5" in MODEL_NAME.lower() else ""
#         query_emb = await asyncio.to_thread(
#             model.encode, [f"{prefix}{processed_query}"], convert_to_numpy=True, normalize_embeddings=True
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
#                 if similarity >= SIMILARITY_THRESHOLD:
#                     results.append({
#                         'chunk_id': meta['chunk_id'],
#                         'source': meta['source'],
#                         'text': chunk['text'],
#                         'metadata': chunk['metadata'],
#                         'distance': float(dist),
#                         'similarity': similarity
#                     })
        
#         logger.info(f"Retrieved {len(results)} chunks for query")
#         return results
#     except Exception as e:
#         logger.error(f"Retrieval failed: {e}")
#         sys.exit(1)

# async def main():
#     parser = argparse.ArgumentParser(description="Retrieve relevant chunks for a query")
#     parser.add_argument('--query', required=True, help="Query string")
#     parser.add_argument('--top_k', type=int, default=TOP_K, help="Number of chunks to retrieve")
#     args = parser.parse_args()
    
#     try:
#         results = await retrieve(args.query, args.top_k)
#         for i, res in enumerate(results, 1):
#             logger.info(f"Result {i}: {res['source']} (Similarity: {res['similarity']:.4f}, Distance: {res['distance']:.4f})")
#             logger.info(f"Text: {res['text'][:100]}...")
#             logger.info(f"Metadata: {res['metadata']}\n")
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")
#         sys.exit(1)

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










# import os
# import json
# import numpy as np
# import faiss
# import logging
# import asyncio
# import sys
# import yaml
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# import argparse
# import re

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def load_config():
#     """Load configuration from rag_config.yaml."""
#     try:
#         CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "rag_config.yaml")
#         logger.info(f"Loading config from: {CONFIG_PATH}")
#         with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
#         logger.info(f"Config loaded successfully")
#         return config
#     except Exception as e:
#         logger.error(f"Failed to load rag_config.yaml: {e}")
#         sys.exit(1)

# # Load configuration
# config = load_config()

# # Define paths and settings from config
# BASE_DIR = Path(__file__).resolve().parent.parent
# INDEX_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['index_file'])
# INDEX_METADATA_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['metadata_file'])
# CHUNKS_FILE = os.path.join(config['data']['processed_dir'], config['data']['preprocessing']['output_file'])
# MODEL_NAME = config['retrieval']['embedding_model']
# TOP_K = config['retrieval']['top_k']
# SIMILARITY_THRESHOLD = config['retrieval']['similarity_threshold']

# # Debug logging for paths and config
# logger.info(f"BASE_DIR: {BASE_DIR}")
# logger.info(f"INDEX_FILE: {INDEX_FILE}")
# logger.info(f"INDEX_METADATA_FILE: {INDEX_METADATA_FILE}")
# logger.info(f"CHUNKS_FILE: {CHUNKS_FILE}")
# logger.info(f"MODEL_NAME: {MODEL_NAME}")
# logger.info(f"TOP_K: {TOP_K}")
# logger.info(f"SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")

# async def load_index_and_metadata():
#     """Load FAISS index and metadata asynchronously."""
#     try:
#         logger.info("Loading FAISS index and metadata...")
        
#         # Debug: Check if files exist
#         logger.info(f"Checking if INDEX_FILE exists: {os.path.exists(INDEX_FILE)}")
#         logger.info(f"Checking if INDEX_METADATA_FILE exists: {os.path.exists(INDEX_METADATA_FILE)}")
#         logger.info(f"Checking if CHUNKS_FILE exists: {os.path.exists(CHUNKS_FILE)}")
        
#         index = await asyncio.to_thread(faiss.read_index, str(INDEX_FILE))
#         logger.info(f"FAISS index loaded - ntotal: {index.ntotal}, d: {index.d}")
        
#         with open(INDEX_METADATA_FILE, 'r') as f:
#             index_metadata = json.load(f)
#         logger.info(f"Index metadata loaded - length: {len(index_metadata)}")
        
#         with open(CHUNKS_FILE, 'r') as f:
#             chunks = {json.loads(line)['id']: json.loads(line) for line in f}
#         logger.info(f"Chunks loaded - total chunks: {len(chunks)}")
        
#         # Debug: Show sample chunk IDs
#         sample_chunk_ids = list(chunks.keys())[:5]
#         logger.info(f"Sample chunk IDs: {sample_chunk_ids}")
        
#         # Debug: Show sample metadata
#         sample_metadata = index_metadata[:3] if len(index_metadata) > 0 else []
#         logger.info(f"Sample metadata entries: {sample_metadata}")
        
#         logger.info("Index and metadata loaded successfully")
#         return index, index_metadata, chunks
#     except Exception as e:
#         logger.error(f"Failed to load index or metadata: {e}")
#         sys.exit(1)

# def preprocess_query(query):
#     """Preprocess query to focus on key terms for diverse movie data."""
#     logger.info(f"Starting query preprocessing for: '{query}'")
    
#     filler_words = {
#         'best', 'top', 'good', 'great', 'movie', 'film', 'kmovie', 'kdrama', 'anime', 'manga',
#         'bollywood', 'hollywood', 'on', 'about', 'what', 'is', 'are', 'the', 'a', 'an',
#         'in', 'of', 'for', 'to', 'and', 'with', 'how', 'why', 'which'
#     }
#     query = query.lower()
#     logger.info(f"Lowercased query: '{query}'")
    
#     # Split into words, preserving phrases like "train to busan"
#     words = re.findall(r'\b[\w\s-]+\b', query)  # Match words and phrases
#     logger.info(f"Words found: {words}")
    
#     key_terms = [w.strip() for w in words if w.strip() not in filler_words and len(w.strip()) > 3]
#     logger.info(f"Key terms after filtering: {key_terms}")
    
#     processed = ' '.join(key_terms) if key_terms else query
#     logger.info(f"Final preprocessed query: '{processed}'")
#     return processed

# async def retrieve(query, top_k=TOP_K):
#     """Retrieve top-k chunks asynchronously."""
#     try:
#         logger.info(f"Starting retrieval for query: '{query}' with top_k: {top_k}")
        
#         # Reload config to ensure the latest MODEL_NAME is used
#         config = load_config()
#         global MODEL_NAME, SIMILARITY_THRESHOLD
#         MODEL_NAME = config['retrieval']['embedding_model']
#         SIMILARITY_THRESHOLD = config['retrieval']['similarity_threshold']
#         EMBEDDING_DIMENSION = config['retrieval']['embedding_dimension']
        
#         logger.info(f"Updated MODEL_NAME: {MODEL_NAME}")
#         logger.info(f"Updated SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
#         logger.info(f"EMBEDDING_DIMENSION: {EMBEDDING_DIMENSION}")
        
#         # Preprocess query
#         processed_query = preprocess_query(query)
#         logger.info(f"Original query: {query}")
#         logger.info(f"Processed query: {processed_query}")
        
#         # Load model and encode query with proper prefix
#         logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
#         model = SentenceTransformer(MODEL_NAME)
        
#         # Add "query: " prefix for E5 models
#         prefix = "query: " if "e5" in MODEL_NAME.lower() else ""
#         prefixed_query = f"{prefix}{processed_query}"
#         logger.info(f"Query with prefix: '{prefixed_query}'")
        
#         query_emb = await asyncio.to_thread(
#             model.encode, [prefixed_query], convert_to_numpy=True, normalize_embeddings=True
#         )
#         query_emb = query_emb.astype(np.float32)
#         logger.info(f"Query embedding shape: {query_emb.shape}")
#         logger.info(f"Query embedding dtype: {query_emb.dtype}")
#         logger.info(f"Query embedding norm: {np.linalg.norm(query_emb)}")
#         logger.info(f"Query embedding sample values: {query_emb[0][:5]}")
        
#         # Load index and search
#         index, index_metadata, chunks = await load_index_and_metadata()
        
#         logger.info(f"About to search index with:")
#         logger.info(f"  - Index ntotal: {index.ntotal}")
#         logger.info(f"  - Index dimension: {index.d}")
#         logger.info(f"  - Query embedding dimension: {query_emb.shape[1]}")
#         logger.info(f"  - Requesting top_k: {top_k}")
        
#         distances, indices = await asyncio.to_thread(index.search, query_emb, top_k)
        
#         logger.info(f"Search results: indices={indices}, distances={distances}")
#         logger.info(f"Indices shape: {indices.shape}, Distances shape: {distances.shape}")
#         logger.info(f"Raw indices: {indices[0]}")
#         logger.info(f"Raw distances: {distances[0]}")
        
#         results = []
#         for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
#             logger.info(f"Processing result {i+1}: idx={idx}, dist={dist}")
            
#             if idx >= 0:  # Ensure valid index
#                 logger.info(f"Valid index {idx}, getting metadata...")
                
#                 if idx >= len(index_metadata):
#                     logger.warning(f"Index {idx} is out of range for metadata (length: {len(index_metadata)})")
#                     continue
                    
#                 meta = index_metadata[idx]
#                 logger.info(f"Metadata for index {idx}: {meta}")
                
#                 chunk = chunks.get(meta['chunk_id'])
#                 if chunk is None:
#                     logger.warning(f"Chunk with ID {meta['chunk_id']} not found in chunks file")
#                     continue
                    
#                 similarity = float(1 - dist)  # Convert distance to similarity
#                 logger.info(f"Calculated similarity: {similarity} (distance: {dist})")
#                 logger.info(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
                
#                 if similarity >= SIMILARITY_THRESHOLD:
#                     logger.info(f"Similarity {similarity} >= threshold {SIMILARITY_THRESHOLD}, adding to results")
#                     result = {
#                         'chunk_id': meta['chunk_id'],
#                         'source': meta['source'],
#                         'text': chunk['text'],
#                         'metadata': chunk['metadata'],
#                         'distance': float(dist),
#                         'similarity': similarity
#                     }
#                     results.append(result)
#                     logger.info(f"Added result: chunk_id={result['chunk_id']}, source={result['source']}")
#                 else:
#                     logger.info(f"Similarity {similarity} < threshold {SIMILARITY_THRESHOLD}, skipping")
#             else:
#                 logger.info(f"Invalid index {idx}, skipping")
        
#         logger.info(f"Final results count: {len(results)}")
#         logger.info(f"Retrieved {len(results)} chunks for query")
#         return results
#     except Exception as e:
#         logger.error(f"Retrieval failed: {e}")
#         logger.error(f"Exception type: {type(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         sys.exit(1)

# async def main():
#     parser = argparse.ArgumentParser(description="Retrieve relevant chunks for a query")
#     parser.add_argument('--query', required=True, help="Query string")
#     parser.add_argument('--top_k', type=int, default=TOP_K, help="Number of chunks to retrieve")
#     args = parser.parse_args()
    
#     logger.info(f"Starting main with query: '{args.query}', top_k: {args.top_k}")
    
#     try:
#         results = await retrieve(args.query, args.top_k)
        
#         logger.info(f"Main received {len(results)} results")
        
#         if len(results) == 0:
#             logger.warning("No results found!")
#             logger.info("This could be due to:")
#             logger.info("1. Similarity threshold too high")
#             logger.info("2. No matching content in the index")
#             logger.info("3. Query preprocessing removing too many terms")
#             logger.info("4. Embedding model mismatch")
        
#         for i, res in enumerate(results, 1):
#             logger.info(f"Result {i}: {res['source']} (Similarity: {res['similarity']:.4f}, Distance: {res['distance']:.4f})")
#             logger.info(f"Text: {res['text'][:100]}...")
#             logger.info(f"Metadata: {res['metadata']}\n")
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")
#         logger.error(f"Exception type: {type(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         sys.exit(1)

# if __name__ == "__main__":
#     asyncio.run(main())