import logging
from pathlib import Path
import yaml
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_system"
CONFIG_PATH = BASE_DIR / "config" / "generation_config.yaml"
log_dir = BASE_DIR / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "rag_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ResponseRefiner:
    """Class to refine LLM responses for accuracy and relevance."""
    
    def __init__(self):
        """Initialize with embedding model and configuration."""
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
            logger.info("ResponseRefiner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ResponseRefiner: {e}")
            raise
    
    def refine_response(self, response: str, query: str, contexts: Dict[str, List]) -> str:
        """Refine the LLM response by validating against contexts."""
        try:
            # Split response into sentences
            sentences = [s.strip() for s in response.split('. ') if s.strip()]
            if not sentences:
                return response
            
            # Embed query and response sentences
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            sentence_embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True)
            
            # Combine all contexts (YouTube, web, local)
            all_context_texts = []
            for ctx_type, ctx_list in contexts.items():
                texts = [ctx.get('text', ctx.get('content', ctx)) if isinstance(ctx, dict) else ctx for ctx in ctx_list]
                all_context_texts.extend(texts)
            
            if not all_context_texts:
                logger.warning("No context available for response refinement")
                return response
            
            # Embed contexts
            context_embeddings = self.embedding_model.encode(all_context_texts, convert_to_numpy=True)
            
            # Calculate similarity between response sentences and contexts
            refined_sentences = []
            for i, (sentence, sent_embedding) in enumerate(zip(sentences, sentence_embeddings)):
                similarities = cosine_similarity([sent_embedding], context_embeddings)[0]
                max_similarity = np.max(similarities)
                
                # Keep sentences with high similarity to context or query
                query_similarity = cosine_similarity([sent_embedding], [query_embedding])[0][0]
                if max_similarity >= 0.6 or query_similarity >= 0.7:
                    refined_sentences.append(sentence)
                else:
                    logger.debug(f"Filtered out sentence due to low relevance: {sentence}")
            
            refined_response = ". ".join(refined_sentences) + ("." if refined_sentences else "")
            logger.info(f"Refined response from {len(sentences)} to {len(refined_sentences)} sentences")
            return refined_response if refined_response else response
        except Exception as e:
            logger.error(f"Response refinement failed: {e}")
            return response
