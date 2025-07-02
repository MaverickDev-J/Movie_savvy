import logging
from pathlib import Path
import yaml
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_system"
CONFIG_PATH = BASE_DIR / "config" / "processing_config.yaml"
INDEX_PATH = BASE_DIR / "output" / "index" / "faiss_index.bin"
METADATA_PATH = BASE_DIR / "output" / "index" / "metadata.json"

# Logging setup
log_dir = BASE_DIR / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "rag_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ChunkEmbeddingManager:
    """Class to manage chunk embeddings and FAISS index."""
    
    def __init__(self):
        """Initialize embedding model and FAISS index."""
        try:
            self.config = self.load_config()
            self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
            self.dimension = self.config.get('embedding_dimension', 1024)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            logger.info("ChunkEmbeddingManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChunkEmbeddingManager: {e}")
            raise
    
    def load_config(self) -> Dict:
        """Load processing configuration from YAML file."""
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("Processing configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load processing_config.yaml: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for a list of chunks."""
        try:
            texts = [chunk.get('text', '') for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            return np.array([])
    
    def add_to_index(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the FAISS index."""
        try:
            self.index.add(embeddings)
            self.metadata.extend(chunks)
            logger.info(f"Added {len(chunks)} chunks to FAISS index")
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}")
            raise
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, str(INDEX_PATH))
            with open(METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved FAISS index to {INDEX_PATH} and metadata to {METADATA_PATH}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded FAISS index from {INDEX_PATH} and metadata from {METADATA_PATH}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def search_index(self, query: str, top_k: int = 8) -> List[Dict]:
        """Search the FAISS index for top-k relevant chunks."""
        try:
            query_embedding = self.embed_chunks([{'text': query}])[0]
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    chunk = self.metadata[idx].copy()
                    chunk['similarity'] = 1.0 - distance / 2.0  # Convert L2 distance to similarity
                    results.append(chunk)
            logger.info(f"Retrieved {len(results)} chunks from FAISS index for query")
            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
