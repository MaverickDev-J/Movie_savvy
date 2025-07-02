import yaml
import logging
import re
from pathlib import Path
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_system"
CONFIG_PATH = BASE_DIR / "config" / "processing_config.yaml"
PROMPT_PATH = BASE_DIR / "generation" / "prompt_templates" / "intent_analysis_prompt.txt"

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

class EnhancedIntentClassifier:
    """Class for analyzing query intent and determining YouTube content necessity."""
    
    def __init__(self):
        """Initialize classifier with embedding model and configuration."""
        try:
            self.config = self.load_config()
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
            self.model = AutoModelForCausalLM.from_pretrained(
                'mistralai/Mistral-7B-Instruct-v0.3',
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.eval()
            logger.info("EnhancedIntentClassifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedIntentClassifier: {e}")
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
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for the input query."""
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            logger.info(f"Generated embedding for query: {query[:50]}...")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []
    
    def build_intent_prompt(self, query: str) -> str:
        """Build the prompt for intent analysis using the template."""
        try:
            with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
                template = f.read()
            prompt = template.format(query=query)
            logger.debug(f"Built intent prompt for query: {query[:50]}...")
            return prompt
        except Exception as e:
            logger.error(f"Failed to build intent prompt: {e}")
            return ""
    
    def _extract_movie_names(self, query: str) -> List[str]:
        """Extract potential movie names from the query."""
        words = re.findall(r'\b[A-Z][a-zA-Z0-9\s-]+\b', query)
        return [word.strip() for word in words if len(word.strip()) > 3]
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse structured LLM response into a dictionary."""
        try:
            response_marker = "### Response:"
            if response_marker in response:
                json_str = response[response.find(response_marker) + len(response_marker):].strip()
                return yaml.safe_load(json_str) or {}
            return {}
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {}

    def analyze_query_intent(self, query: str) -> Dict:
        """Analyze query intent to determine if YouTube content is needed."""
        try:
            query_lower = query.lower()
            intent_patterns = self.config.get('intent_patterns', {})
            high_confidence = intent_patterns.get('high_youtube_confidence', [])
            medium_confidence = intent_patterns.get('medium_youtube_confidence', [])
            search_optimization = intent_patterns.get('youtube_search_optimization', {})
            
            needs_youtube = False
            confidence_score = 0.0
            query_type = 'other'
            expected_video_types = []
            
            for keyword in high_confidence:
                if keyword in query_lower:
                    needs_youtube = True
                    confidence_score = max(confidence_score, 0.9)
                    if keyword in ['trailer']:
                        query_type = 'trailer_request'
                        expected_video_types.append('trailer')
                    elif keyword in ['review', 'analysis']:
                        query_type = 'review_analysis'
                        expected_video_types.extend(['review', 'analysis'])
                    elif keyword in ['behind the scenes']:
                        query_type = 'behind_scenes'
                        expected_video_types.append('behind_scenes')
                    elif keyword in ['comparison']:
                        query_type = 'comparison'
                        expected_video_types.append('comparison')
                    elif keyword in ['reaction']:
                        query_type = 'reaction'
                        expected_video_types.append('reaction')
                    elif keyword in ['fan theories', 'fan theory', 'discussion', 'discussions']:
                        query_type = 'fan_discussion'
                        expected_video_types.extend(['fan theory', 'discussion'])
                        confidence_score = max(confidence_score, 0.85)
                    elif keyword in ['youtuber', 'youtubers', 'youtube suggestions', 'youtube recommendation', 'famous youtubers']:
                        query_type = 'youtuber_recommendation'
                        expected_video_types.extend(['recommendation', 'top movies'])
                        confidence_score = max(confidence_score, 0.9)
            
            if not needs_youtube:
                for keyword in medium_confidence:
                    if keyword in query_lower:
                        needs_youtube = True
                        confidence_score = max(confidence_score, 0.6)
                        query_type = 'explanation' if query_type == 'other' else query_type
                        expected_video_types.extend(['explanation', 'breakdown'])
            
            youtube_search_terms = [query]
            if query_type == 'trailer_request':
                youtube_search_terms.append(f"{query} official trailer")
            elif query_type == 'review_analysis':
                youtube_search_terms.append(f"{query} review analysis")
            elif query_type == 'behind_scenes':
                youtube_search_terms.append(f"{query} behind the scenes")
            elif query_type == 'fan_discussion':
                youtube_search_terms.append(f"{query} fan theories discussion")
            elif query_type == 'youtuber_recommendation':
                youtube_search_terms.append(f"{query} youtuber recommendations")
            
            query_embedding = self.embed_query(query)
            
            prompt = self.build_intent_prompt(query)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed_response = self._parse_llm_response(response)
            
            result = {
                "needs_youtube": needs_youtube or parsed_response.get('needs_youtube', False),
                "query_type": parsed_response.get('query_type', query_type),
                "youtube_search_terms": youtube_search_terms or parsed_response.get('youtube_search_terms', []),
                "confidence_score": max(confidence_score, parsed_response.get('confidence_score', 0.0)),
                "query_embedding": query_embedding,
                "expected_video_types": list(set(expected_video_types + parsed_response.get('expected_video_types', [])))
            }
            
            logger.info(f"Intent analysis result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "needs_youtube": False,
                "query_type": "other",
                "youtube_search_terms": [],
                "confidence_score": 0.0,
                "query_embedding": [],
                "expected_video_types": []
            }