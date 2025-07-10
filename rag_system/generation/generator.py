import torch
import sys
from pathlib import Path
import logging
import yaml

# ----- paths ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
sys.path.insert(0, str(LITGPT_DIR))

# ----- configs ----------------------------------------------------------
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"

with open(CONFIG_PATH, "r") as f:
    rag_config = yaml.safe_load(f)
with open(MODEL_CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)

# ----- imports ----------------------------------------------------------
try:
    from litgpt import GPT, Config, Tokenizer
    from litgpt.utils import load_checkpoint
    from litgpt.generate.base import generate
    import lightning.fabric as fabric
    LITGPT_AVAILABLE = True
except ImportError:
    LITGPT_AVAILABLE = False
    GPT = Config = Tokenizer = load_checkpoint = generate = fabric = None

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
MODEL_DIR = BASE_DIR / model_config["fine_tuned_model"]["path"]
PROMPT_DIR = BASE_DIR / rag_config["generation"]["prompt"]["template_dir"]
DEFAULT_PROMPT_FILE = "rag_prompt.txt"
REDDIT_PROMPT_FILE = "rag_reddit_prompt.txt"

# -----------------------------------------------------------------------
class Generator:
    """Wraps LitGPT model + tokenizer and builds prompts."""

    def __init__(self):
        logger.info("Initializing Generator …")
        self.device = (model_config["performance"]["device_map"] if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
        self.model, self.tokenizer = self._load_litgpt_model()
        self.default_prompt_template = self._load_prompt_template(DEFAULT_PROMPT_FILE)
        self.reddit_prompt_template = self._load_prompt_template(REDDIT_PROMPT_FILE)
        logger.info("Generator ready ✔")

    # ---------------- private helpers ----------------------------------
    def _load_litgpt_model(self):
        if not LITGPT_AVAILABLE:
            logger.error("LitGPT not available – using mock")
            return self._mock_model(), None
        try:
            logger.info(f"Loading LitGPT model from {MODEL_DIR}")
            config_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["model_config"]
            if not config_path.exists():
                logger.error(f"Config not found: {config_path}")
                return self._mock_model(), None
            tokenizer = Tokenizer(MODEL_DIR)
            logger.info("Tokenizer loaded")
            with torch.device(self.device):
                model = GPT(Config.from_file(config_path))
            # checkpoint
            ckpt_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["checkpoint"]
            if not ckpt_path.exists():
                logger.error(f"Checkpoint not found: {ckpt_path}")
                return self._mock_model(), None
            fabric_instance = fabric.Fabric(devices=model_config["loading"]["fabric_devices"],
                                            accelerator=model_config["loading"]["fabric_accelerator"],
                                            precision=model_config["training"]["precision"])
            model = fabric_instance.setup(model)
            load_checkpoint(fabric_instance, model, ckpt_path)
            model.to(self.device).eval()
            # kv‑cache
            model.set_kv_cache(batch_size=1, device=self.device)
            logger.info("KV cache initialized")
            # lora weights (optional)
            lora_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["lora_weights"]
            if lora_path.exists():
                logger.info("Applying LoRA weights...")
                try:
                    lora_state = torch.load(lora_path, map_location=self.device)
                    model.load_state_dict(lora_state, strict=False)
                    logger.info("LoRA weights applied successfully")
                except Exception as e:
                    logger.warning(f"Failed to apply LoRA weights: {e}")
            return model, tokenizer
        except Exception as exc:
            logger.exception("Model load failed, using mock", exc_info=exc)
            return self._mock_model(), None

    def _mock_model(self):
        logger.warning("Using mock model...")
        class Mock:
            def generate(self, *_, **__):
                return "Mock response – model not loaded"
        return Mock()

    def _load_prompt_template(self, filename):
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
        if filename == REDDIT_PROMPT_FILE:
            return """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to deliver concise, engaging answers in a geeky, conversational tone. Answer the query using official data for facts and Reddit discussions for fan opinions, without inventing details. If information is limited, say so clearly and provide a brief, factual response.

    Instruction: {query}

    Official Data: {vector_context}

    Community Discussion: {reddit_context}

    Response Guidelines:
    - Use official data for plot, cast, ratings, or technical details.
    - Use Reddit discussion for fan reactions or opinions, clearly labeling them as "fans say" or "Reddit users mention."
    - Keep responses concise (2–4 sentences) unless a detailed breakdown is requested.
    - If no relevant data is available, state: "Limited info available, but here's a general take."
    - Maintain a fun, movie-enthusiast tone, blending facts and fan insights naturally.
    - Avoid lists unless the query explicitly asks for them.

    ### Response:"""
        else:
            return """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to provide concise, engaging, and accurate answers in a conversational, geeky tone. Use only the provided context to answer the query. If the context is insufficient, say so clearly and provide a brief, factual response based on general entertainment knowledge.

    Instruction: {query}

    Context: {context}

    Response Guidelines:
    - Answer directly using the context, avoiding invented details.
    - Keep responses short (2–4 sentences) unless a detailed explanation is requested.
    - Use a fun, movie-enthusiast tone, like chatting with a fellow fan.
    - If context is missing or irrelevant, state: "Not enough context for details, but here's what I can share."
    - Format as a natural narrative, not a list, unless the query asks for one.

    ### Response:"""

    # ---------------- public API ----------------------------------------
    def generate(
        self,
        query: str,
        vector_context: list[str],
        reddit_context: list[str] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        """Build prompt (docs) and call LitGPT."""
        max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
        temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
        top_k = top_k or rag_config["generation"]["parameters"].get("top_k", 50)

        prompt = self._build_prompt(
            vector_context=vector_context,
            reddit_context=reddit_context,
            current_query=query,
            use_reddit=bool(reddit_context)
        )

        logger.debug("Prompt:\n%s", prompt)
        if not (self.tokenizer and self.model):
            return "[ERROR] Model/tokenizer missing."

        # Tokenize
        encoded = self.tokenizer.encode(prompt, device=self.device)
        block = getattr(self.model.config, "block_size", 2048)
        avail = block - encoded.size(0)
        max_new_tokens = min(max_new_tokens, max(avail, 1))

        try:
            ids = generate(model=self.model, prompt=encoded, max_returned_tokens=encoded.size(0) + max_new_tokens, temperature=temperature, top_k=top_k, eos_id=self.tokenizer.eos_id)
            text = self.tokenizer.decode(ids)
            # Strip everything before ### Response: and clean up
            response_marker = "### Response:"
            if response_marker in text:
                answer = text[text.find(response_marker) + len(response_marker):].strip()
            else:
                answer = text[len(prompt):].strip()
            return answer
        except Exception as exc:
            logger.exception("Generation failed", exc_info=exc)
            return "[ERROR] Generation failed."

    def _build_prompt(self, vector_context: list[str], reddit_context: list[str] | None, current_query: str, use_reddit: bool) -> str:
        """Build the prompt using the appropriate template."""
        template = self.reddit_prompt_template if use_reddit else self.default_prompt_template
        context = "\n".join(vector_context) if vector_context else "No official context available."
        reddit_context_str = "\n".join(reddit_context) if reddit_context else "No community discussion available."
        return template.format(
            query=current_query,
            vector_context=context,
            reddit_context=reddit_context_str if use_reddit else context
        )

    def get_model_info(self):
        return {
            "litgpt_available": LITGPT_AVAILABLE,
            "device": str(self.device),
            "model_dir": str(MODEL_DIR),
            "prompt_dir": str(PROMPT_DIR),
            "default_prompt_file": DEFAULT_PROMPT_FILE,
            "reddit_prompt_file": REDDIT_PROMPT_FILE,
            "litgpt_dir": str(LITGPT_DIR),
            "litgpt_dir_exists": LITGPT_DIR.exists(),
            "model_type": type(self.model).__name__ if hasattr(self.model, "__class__") else "Unknown",
            "load_checkpoint_available": load_checkpoint is not None,
            "generate_function_available": generate is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }

if __name__ == "__main__":
    logger.info("Starting generator initialization...")
    try:
        generator = Generator()
        model_info = generator.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        logger.info("Generator initialized successfully!")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")










