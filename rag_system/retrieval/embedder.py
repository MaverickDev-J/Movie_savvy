import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, "processed_chunks.jsonl")
EMBEDDINGS_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings")
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.json")

# Ensure embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def load_chunks():
    """Load processed chunks from JSONL file."""
    chunks = []
    try:
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading chunks"):
                chunk = json.loads(line.strip())
                chunks.append(chunk)
        logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
        return chunks
    except FileNotFoundError:
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        raise
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        raise

def generate_and_save_embeddings_by_source(chunks, model_name='all-MiniLM-L6-v2'):
    """Generate and save embeddings grouped by source file for memory efficiency."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        model = SentenceTransformer(model_name, device=device)
        
        # Group chunks by source file
        source_to_chunks = defaultdict(list)
        for i, chunk in enumerate(chunks):
            source = chunk['metadata']['source']
            source_to_chunks[source].append((i, chunk))
        
        embedding_paths = {}
        chunk_to_embedding_info = {}
        
        # Process each source file separately
        for source, chunk_list in tqdm(source_to_chunks.items(), desc="Processing sources"):
            logger.info(f"Processing {len(chunk_list)} chunks from {source}")
            
            # Extract texts for this source
            texts = [chunk['text'] for _, chunk in chunk_list]
            
            # Generate embeddings for this source
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
            
            # Save embeddings
            output_file = os.path.join(EMBEDDINGS_DIR, f"{source.replace('.json', '')}.npy")
            np.save(output_file, embeddings)
            embedding_paths[source] = output_file
            
            # Store embedding info for metadata update
            for idx, (original_idx, chunk) in enumerate(chunk_list):
                chunk_to_embedding_info[chunk['id']] = {
                    'embedding_file': output_file,
                    'embedding_index': idx,
                    'source': source
                }
            
            logger.info(f"Saved {len(embeddings)} embeddings to {output_file}")
        
        return embedding_paths, chunk_to_embedding_info
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def update_metadata(chunk_to_embedding_info):
    """Update metadata with embedding file paths and indices."""
    try:
        # Load existing metadata
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Update metadata with embedding information
        for item in tqdm(metadata, desc="Updating metadata"):
            item['embedding_info'] = {}
            
        # Create a more efficient structure
        embedding_metadata = {
            'embedding_files': {},
            'chunk_mappings': chunk_to_embedding_info
        }
        
        # Get unique embedding files
        embedding_files = set(info['embedding_file'] for info in chunk_to_embedding_info.values())
        for i, file_path in enumerate(embedding_files):
            embedding_metadata['embedding_files'][f'file_{i}'] = file_path
        
        # Save updated metadata with embedding info
        updated_metadata = {
            'items': metadata,
            'embeddings': embedding_metadata
        }
        
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(updated_metadata, f, indent=2)
            
        logger.info(f"Updated metadata with embedding information for {len(chunk_to_embedding_info)} chunks")
        
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {METADATA_FILE}")
        raise
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")
        raise

def main():
    """Main function to generate and save embeddings."""
    try:
        # Check if required files exist
        if not os.path.exists(CHUNKS_FILE):
            logger.error(f"Required file not found: {CHUNKS_FILE}")
            logger.error("Please run the data processing script first.")
            return
        
        # Load chunks
        chunks = load_chunks()
        
        if not chunks:
            logger.warning("No chunks found to process")
            return
        
        # Generate and save embeddings by source
        embedding_paths, chunk_to_embedding_info = generate_and_save_embeddings_by_source(chunks)
        
        logger.info(f"Generated embeddings for {len(embedding_paths)} source files")
        
        # Update metadata
        update_metadata(chunk_to_embedding_info)
        
        logger.info("Embedding generation completed successfully")
        logger.info(f"Embedding files saved in: {EMBEDDINGS_DIR}")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

if __name__ == "__main__":
    main()