import json
import os
import numpy as np
import faiss
import logging
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "processed", "embeddings")
METADATA_FILE = os.path.join(BASE_DIR, "data", "processed", "metadata.json")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "processed", "processed_chunks.jsonl")
INDEX_DIR = os.path.join(BASE_DIR, "retrieval", "index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.bin")
INDEX_METADATA_FILE = os.path.join(INDEX_DIR, "index_metadata.json")

# Ensure index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

def load_embeddings():
    """Load all .npy embedding files and their corresponding chunk IDs."""
    embeddings = []
    chunk_ids = []
    source_files = []

    logger.info(f"Loading embeddings from {EMBEDDINGS_DIR}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line.strip()) for line in f]
    
    # Create a mapping of source to chunk IDs
    source_to_chunks = {}
    for chunk in chunks:
        source = chunk['metadata']['source']
        if source not in source_to_chunks:
            source_to_chunks[source] = []
        source_to_chunks[source].append(chunk['id'])

    # Load embeddings for each source
    for source_file in os.listdir(EMBEDDINGS_DIR):
        if source_file.endswith('.npy'):
            source = source_file.replace('.npy', '.json')
            emb_path = os.path.join(EMBEDDINGS_DIR, source_file)
            source_emb = np.load(emb_path).astype(np.float32)
            embeddings.append(source_emb)
            
            if source in source_to_chunks:
                chunk_ids.extend(source_to_chunks[source])
                source_files.extend([source] * len(source_to_chunks[source]))
                logger.info(f"Loaded {len(source_to_chunks[source])} embeddings from {source_file}")
            else:
                logger.warning(f"No chunks found for source {source}")

    if not embeddings:
        raise ValueError("No embeddings loaded")

    embeddings = np.vstack(embeddings)
    logger.info(f"Total embeddings shape: {embeddings.shape}")
    return embeddings, chunk_ids, source_files

def build_faiss_index(embeddings):
    """Build a FAISS index for the embeddings."""
    try:
        dim = embeddings.shape[1]
        logger.info(f"Building FAISS index with dimension {dim}")
        
        # Use FlatL2 index for exact search
        index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to the index
        logger.info("Adding embeddings to FAISS index")
        index.add(embeddings)
        
        logger.info(f"Index contains {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise

def save_index(index, chunk_ids, source_files):
    """Save the FAISS index and index metadata."""
    try:
        # Save FAISS index
        faiss.write_index(index, INDEX_FILE)
        logger.info(f"Saved FAISS index to {INDEX_FILE}")

        # Create index metadata mapping index IDs to chunk IDs and sources
        index_metadata = [
            {"index_id": i, "chunk_id": chunk_id, "source": source}
            for i, (chunk_id, source) in enumerate(zip(chunk_ids, source_files))
        ]

        # Save index metadata
        with open(INDEX_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(index_metadata, f, indent=2)
        logger.info(f"Saved index metadata to {INDEX_METADATA_FILE}")

        # Update metadata.json with index_id
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata_content = json.load(f)

        # Handle both old and new metadata formats
        if isinstance(metadata_content, dict) and 'items' in metadata_content:
            # New format with 'items' key
            metadata = metadata_content['items']
            logger.info("Using new metadata format with 'items' key")
        elif isinstance(metadata_content, list):
            # Old format - direct list
            metadata = metadata_content
            logger.info("Using old metadata format (direct list)")
        else:
            raise ValueError("Unexpected metadata format")

        # Create a mapping of chunk_id to index_id
        chunk_to_index = {item["chunk_id"]: item["index_id"] for item in index_metadata}

        # Update metadata with index_id for chunks - OPTIMIZED VERSION
        logger.info("Building item_id to index_ids mapping...")
        
        # Create a mapping from item_id to list of index_ids
        item_to_indices = {}
        
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing chunks"):
                chunk = json.loads(line.strip())
                chunk_id = chunk['id']
                item_id = chunk['metadata']['item_id']
                
                # Get the index_id for this chunk
                if chunk_id in chunk_to_index:
                    index_id = chunk_to_index[chunk_id]
                    
                    # Add to item mapping
                    if item_id not in item_to_indices:
                        item_to_indices[item_id] = []
                    item_to_indices[item_id].append(index_id)

        # Now update metadata items efficiently
        for item in tqdm(metadata, desc="Updating metadata"):
            if not isinstance(item, dict):
                logger.error(f"Expected dict but got {type(item)}: {item}")
                continue
                
            item_id = item['item_id']
            item['index_ids'] = item_to_indices.get(item_id, [])

        # Save updated metadata in the same format it was loaded
        if isinstance(metadata_content, dict) and 'items' in metadata_content:
            # Keep the new format
            metadata_content['items'] = metadata
            final_metadata = metadata_content
        else:
            # Keep the old format
            final_metadata = metadata

        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_metadata, f, indent=2)
        logger.info(f"Updated {METADATA_FILE} with index IDs")

    except Exception as e:
        logger.error(f"Error saving index or metadata: {e}")
        raise

def main():
    """Main function to build and save the FAISS index."""
    try:
        # Load embeddings and chunk IDs
        embeddings, chunk_ids, source_files = load_embeddings()

        # Build FAISS index
        index = build_faiss_index(embeddings)

        # Save index and update metadata
        save_index(index, chunk_ids, source_files)

        logger.info("FAISS index building completed successfully")
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        raise

if __name__ == "__main__":
    main()