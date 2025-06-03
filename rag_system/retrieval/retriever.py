import json
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "retrieval" / "index" / "index.bin"
INDEX_METADATA_FILE = BASE_DIR / "retrieval" / "index" / "index_metadata.json"
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "processed_chunks.jsonl"

def load_index_and_metadata():
    index = faiss.read_index(str(INDEX_FILE))
    with open(INDEX_METADATA_FILE, 'r') as f:
        index_metadata = json.load(f)
    with open(CHUNKS_FILE, 'r') as f:
        chunks = {json.loads(line)['id']: json.loads(line) for line in f}
    return index, index_metadata, chunks

def retrieve(query, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    index, index_metadata, chunks = load_index_and_metadata()
    distances, indices = index.search(query_emb, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        meta = index_metadata[idx]
        chunk = chunks[meta['chunk_id']]
        results.append({
            'chunk_id': meta['chunk_id'],
            'source': meta['source'],
            'text': chunk['text'],
            'metadata': chunk['metadata'],
            'distance': float(dist)
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    results = retrieve(args.query, args.top_k)
    for i, res in enumerate(results, 1):
        logger.info(f"Result {i}: {res['source']} (Distance: {res['distance']:.4f})")
        logger.info(f"Text: {res['text'][:100]}...")
        logger.info(f"Metadata: {res['metadata']}\n")

if __name__ == "__main__":
    main()