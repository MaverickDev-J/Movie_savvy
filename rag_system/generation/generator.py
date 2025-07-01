import torch
import sys
from pathlib import Path
import logging
import yaml

# ----- paths ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ----- configs ----------------------------------------------------------
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"

with open(CONFIG_PATH, "r") as f:
    rag_config = yaml.safe_load(f)
with open(MODEL_CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)

# ----- imports ----------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = AutoTokenizer = None

# ----- logging ----------------------------------------------------------
log_dir = BASE_DIR / rag_config["pipeline"]["logging"]["log_dir"]
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / rag_config["pipeline"]["logging"]["log_file"]
logging.basicConfig(
    level=logging.getLevelName(rag_config["pipeline"]["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ----- constants --------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
PROMPT_DIR = BASE_DIR / rag_config["generation"]["prompt"]["template_dir"]
DEFAULT_PROMPT_FILE = "rag_prompt.txt"

# -----------------------------------------------------------------------
class Generator:
    """Wraps Transformers model + tokenizer and builds prompts."""

    def __init__(self):
        logger.info("Initializing Generator with Transformers...")
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
            
        self.model, self.tokenizer = self._load_transformers_model()
        self.default_prompt_template = self._load_prompt_template(DEFAULT_PROMPT_FILE)
        logger.info("Generator ready ✔")

    # ---------------- private helpers ----------------------------------
    def _get_device(self):
        """Determine the best device to use."""
        if model_config.get("performance", {}).get("device_map"):
            device = model_config["performance"]["device_map"]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_transformers_model(self):
        """Load Mistral model using Transformers library."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available – using mock")
            return self._mock_model(), None
            
        try:
            logger.info(f"Loading Mistral model: {MODEL_NAME}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Set pad token if not already set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move to device if not using device_map="auto"
            if self.device != "cuda":
                model = model.to(self.device)
            
            model.eval()
            logger.info("Model loaded successfully")
            
            return model, tokenizer
            
        except Exception as exc:
            logger.exception("Model load failed, using mock", exc_info=exc)
            return self._mock_model(), None

    def _mock_model(self):
        """Create a mock model for fallback."""
        logger.warning("Using mock model...")
        class Mock:
            def generate(self, *_, **__):
                return "Mock response – model not loaded"
        return Mock()

    def _load_prompt_template(self, filename):
        """Load prompt template from file."""
        template_path = PROMPT_DIR / filename
        try:
            if not template_path.exists():
                default_template = self._get_default_template(filename)
                template_path.parent.mkdir(parents=True, exist_ok=True)
                with open(template_path, "w") as f:
                    f.write(default_template)
                logger.info(f"Created prompt template: {template_path}")
                return default_template
                
            with open(template_path, "r") as f:
                template = f.read()
                
            if not template.strip():
                template = self._get_default_template(filename)
                with open(template_path, "w") as f:
                    f.write(template)
                    
            logger.info(f"Prompt template {filename} loaded")
            return template
            
        except Exception as e:
            logger.error(f"Template {filename} loading failed: {e}")
            return self._get_default_template(filename)

    def _get_default_template(self, filename):
        """Enhanced default prompt template."""
        return """You are a specialized entertainment AI assistant with expertise in movies, anime, manga, TV shows, and current entertainment news. You provide accurate, engaging responses in a conversational tone.

QUERY: {query}

LOCAL KNOWLEDGE (from entertainment database):
{local_context}

WEB INSIGHTS (current/recent information):
{web_context}

RESPONSE GUIDELINES:
- Prioritize accuracy and relevance over entertainment value
- If web insights contain current information, integrate it with local knowledge
- For ongoing series/current events, emphasize the most recent information
- Keep responses 2-4 sentences unless detailed explanation is requested
- Use an enthusiastic but informative tone
- If information conflicts, clearly state the most reliable source
- For fan theories, distinguish between confirmed facts and speculation

### Response:"""

    def _format_context(self, context_list: list[str], context_type: str) -> str:
        """Format context with better structure and filtering."""
        if not context_list:
            return f"No {context_type.lower()} available."
        
        # Filter and clean context
        cleaned_context = []
        for i, ctx in enumerate(context_list[:5], 1):  # Limit to top 5
            if ctx and len(ctx.strip()) > 20:  # Filter very short content
                cleaned_ctx = ctx.strip()[:800]  # Limit length
                cleaned_context.append(f"{i}. {cleaned_ctx}")
        
        if not cleaned_context:
            return f"No relevant {context_type.lower()} available."
        
        return "\n".join(cleaned_context)

    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        # Remove excessive special characters and emojis at the end
        import re
        response = re.sub(r'[🎬📺🎮🕹️🎲🎯🎭🎞️📽️\s]{10,}$', '', response)
        
        # Remove repetitive patterns
        response = re.sub(r'(\s+[^\w\s]{2,}){5,}', '', response)
        
        # Ensure proper sentence ending
        response = response.strip()
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response

    # ---------------- public API ----------------------------------------
    def generate(
        self,
        query: str,
        local_context: list[str],
        web_context: list[str],
        query_analysis: dict = None,  # Add this parameter
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        """Enhanced generation with query analysis integration."""
        max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
        temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
        top_k = top_k or rag_config["generation"]["parameters"].get("top_k", 50)

        prompt = self._build_prompt(
            local_context=local_context,
            web_context=web_context,
            current_query=query,
            query_analysis=query_analysis  # Pass query analysis
        )

        logger.debug("Prompt:\n%s", prompt)
        
        if not (self.tokenizer and self.model):
            return "[ERROR] Model/tokenizer missing."

        try:
            # Enhanced tokenization with better truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=3072,  # Increased from 2048
                padding=True
            ).to(self.device)
            
            # Adjust generation parameters based on query type
            if query_analysis and query_analysis.get("query_type") == "opinion":
                temperature = min(temperature * 1.2, 0.9)  # Slightly more creative for opinions
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                )
            
            # Better response extraction
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response_marker = "### Response:"
            if response_marker in full_response:
                answer = full_response[full_response.find(response_marker) + len(response_marker):].strip()
            else:
                answer = full_response[len(prompt):].strip()
            
            # Clean up response
            answer = self._clean_response(answer)
            return answer
            
        except Exception as exc:
            logger.exception("Generation failed", exc_info=exc)
            return "[ERROR] Generation failed."

    def _build_prompt(self, local_context: list[str], web_context: list[str], current_query: str, query_analysis: dict = None) -> str:
        """Enhanced prompt building with better context integration."""
        template = self.default_prompt_template
        
        # Prepare contexts with better formatting
        local_context_str = self._format_context(local_context, "Local Knowledge")
        web_context_str = self._format_context(web_context, "Web Insights")
        
        # Determine context priority based on query analysis
        context_priority = "web" if (query_analysis and query_analysis.get("needs_web_search")) else "local"
        
        return template.format(
            query=current_query,
            local_context=local_context_str,
            web_context=web_context_str,
            context_priority=context_priority
        )

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "device": str(self.device),
            "model_name": MODEL_NAME,
            "prompt_dir": str(PROMPT_DIR),
            "default_prompt_file": DEFAULT_PROMPT_FILE,
            "model_type": type(self.model).__name__ if hasattr(self.model, "__class__") else "Unknown",
            "tokenizer_loaded": self.tokenizer is not None,
            "model_loaded": self.model is not None,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }

if __name__ == "__main__":
    logger.info("Starting generator initialization...")
    try:
        generator = Generator()
        model_info = generator.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
            
        # Test generation
        test_query = "What is an interesting fact about movies?"
        test_local_context = ["Movies have been entertaining audiences for over a century."]
        test_web_context = ["Recent studies show that movies can improve cognitive function."]
        
        logger.info("Testing generation...")
        response = generator.generate(
            query=test_query,
            local_context=test_local_context,
            web_context=test_web_context,
            max_new_tokens=100
        )
        logger.info(f"Test response: {response}")
        logger.info("Generator initialized successfully!")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise