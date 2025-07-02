
import logging
from pathlib import Path
import yaml
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_system"
CONFIG_PATH = BASE_DIR / "config" / "processing_config.yaml"

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

class SemanticContentProcessor:
    """Class to score and filter content based on semantic relevance to the query."""
    
    def __init__(self):
        """Initialize the semantic scorer with embedding model and configuration."""
        try:
            self.config = self.load_config()
            self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
            logger.info("SemanticContentProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticContentProcessor: {e}")
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
    
    def embed_content(self, content: List[str]) -> np.ndarray:
        """Generate embeddings for a list of content chunks."""
        try:
            embeddings = self.embedding_model.encode(content, convert_to_numpy=True, show_progress_bar=False)
            logger.info(f"Generated embeddings for {len(content)} content chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed content: {e}")
            return np.array([])
    
    def score_content(self, query: str, content: List[Dict], min_score: float = 0.5) -> List[Dict]:
        """Score content chunks based on semantic similarity to the query."""
        try:
            if not content:
                logger.warning("No content provided for scoring")
                return []
            
            # Extract text from content (handle both dict and string inputs)
            texts = [c.get('text', c) if isinstance(c, dict) else c for c in content]
            if not texts:
                logger.warning("No valid text found in content")
                return []
            
            # Generate query and content embeddings
            query_embedding = self.embed_content([query])[0]
            content_embeddings = self.embed_content(texts)
            
            if len(content_embeddings) == 0:
                logger.warning("No content embeddings generated")
                return []
            
            # Calculate cosine similarities
            similarities = cosine_similarity([query_embedding], content_embeddings)[0]
            
            # Filter and score content
            scored_content = []
            for i, (item, score) in enumerate(zip(content, similarities)):
                if score >= min_score:
                    scored_item = item.copy() if isinstance(item, dict) else {'text': item}
                    scored_item['similarity'] = float(score)
                    scored_content.append(scored_item)
            
            # Sort by similarity (descending)
            scored_content = sorted(scored_content, key=lambda x: x['similarity'], reverse=True)
            logger.info(f"Scored and filtered {len(content)} chunks to {len(scored_content)} with min_score={min_score}")
            return scored_content
        except Exception as e:
            logger.error(f"Content scoring failed: {e}")
            return []
