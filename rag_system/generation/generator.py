# import torch
# import sys
# from pathlib import Path
# import logging
# import yaml

# # ----- paths ------------------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent.parent
# LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
# sys.path.insert(0, str(LITGPT_DIR))

# # ----- configs ----------------------------------------------------------
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"

# with open(CONFIG_PATH, "r") as f:
#     rag_config = yaml.safe_load(f)
# with open(MODEL_CONFIG_PATH, "r") as f:
#     model_config = yaml.safe_load(f)

# # ----- imports ----------------------------------------------------------
# try:
#     from litgpt import GPT, Config, Tokenizer
#     from litgpt.utils import load_checkpoint
#     from litgpt.generate.base import generate
#     import lightning.fabric as fabric
#     LITGPT_AVAILABLE = True
# except ImportError:
#     LITGPT_AVAILABLE = False
#     GPT = Config = Tokenizer = load_checkpoint = generate = fabric = None

# # ----- logging ----------------------------------------------------------
# log_dir = BASE_DIR / rag_config["pipeline"]["logging"]["log_dir"]
# log_dir.mkdir(parents=True, exist_ok=True)
# log_file = log_dir / rag_config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(rag_config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
# )
# logger = logging.getLogger(__name__)

# # ----- constants --------------------------------------------------------
# MODEL_DIR = BASE_DIR / model_config["fine_tuned_model"]["path"]
# PROMPT_DIR = BASE_DIR / rag_config["generation"]["prompt"]["template_dir"]
# DEFAULT_PROMPT_FILE = "rag_prompt.txt"
# REDDIT_PROMPT_FILE = "rag_reddit_prompt.txt"

# # -----------------------------------------------------------------------
# class Generator:
#     """Wraps LitGPT model + tokenizer and builds prompts."""

#     def __init__(self):
#         logger.info("Initializing Generator …")
#         self.device = (model_config["performance"]["device_map"] if torch.cuda.is_available() else "cpu")
#         logger.info(f"Using device: {self.device}")
#         if torch.cuda.is_available():
#             torch.set_float32_matmul_precision("medium")
#         self.model, self.tokenizer = self._load_litgpt_model()
#         self.default_prompt_template = self._load_prompt_template(DEFAULT_PROMPT_FILE)
#         self.reddit_prompt_template = self._load_prompt_template(REDDIT_PROMPT_FILE)
#         logger.info("Generator ready ✔")

#     # ---------------- private helpers ----------------------------------
#     def _load_litgpt_model(self):
#         if not LITGPT_AVAILABLE:
#             logger.error("LitGPT not available – using mock")
#             return self._mock_model(), None
#         try:
#             logger.info(f"Loading LitGPT model from {MODEL_DIR}")
#             config_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["model_config"]
#             if not config_path.exists():
#                 logger.error(f"Config not found: {config_path}")
#                 return self._mock_model(), None
#             tokenizer = Tokenizer(MODEL_DIR)
#             logger.info("Tokenizer loaded")
#             with torch.device(self.device):
#                 model = GPT(Config.from_file(config_path))
#             # checkpoint
#             ckpt_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["checkpoint"]
#             if not ckpt_path.exists():
#                 logger.error(f"Checkpoint not found: {ckpt_path}")
#                 return self._mock_model(), None
#             fabric_instance = fabric.Fabric(devices=model_config["loading"]["fabric_devices"],
#                                             accelerator=model_config["loading"]["fabric_accelerator"],
#                                             precision=model_config["training"]["precision"])
#             model = fabric_instance.setup(model)
#             load_checkpoint(fabric_instance, model, ckpt_path)
#             model.to(self.device).eval()
#             # kv‑cache
#             model.set_kv_cache(batch_size=1, device=self.device)
#             logger.info("KV cache initialized")
#             # lora weights (optional)
#             lora_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["lora_weights"]
#             if lora_path.exists():
#                 logger.info("Applying LoRA weights...")
#                 try:
#                     lora_state = torch.load(lora_path, map_location=self.device)
#                     model.load_state_dict(lora_state, strict=False)
#                     logger.info("LoRA weights applied successfully")
#                 except Exception as e:
#                     logger.warning(f"Failed to apply LoRA weights: {e}")
#             return model, tokenizer
#         except Exception as exc:
#             logger.exception("Model load failed, using mock", exc_info=exc)
#             return self._mock_model(), None

#     def _mock_model(self):
#         logger.warning("Using mock model...")
#         class Mock:
#             def generate(self, *_, **__):
#                 return "Mock response – model not loaded"
#         return Mock()

#     def _load_prompt_template(self, filename):
#         template_path = PROMPT_DIR / filename
#         try:
#             if not template_path.exists():
#                 default_template = self._get_default_template(filename)
#                 template_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(template_path, "w") as f:
#                     f.write(default_template)
#                 logger.info(f"Created prompt template: {template_path}")
#                 return default_template
#             with open(template_path, "r") as f:
#                 template = f.read()
#             if not template.strip():
#                 template = self._get_default_template(filename)
#                 with open(template_path, "w") as f:
#                     f.write(template)
#             logger.info(f"Prompt template {filename} loaded")
#             return template
#         except Exception as e:
#             logger.error(f"Template {filename} loading failed: {e}")
#             return self._get_default_template(filename)

#     def _get_default_template(self, filename):
#         if filename == REDDIT_PROMPT_FILE:
#             return """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to deliver concise, engaging answers in a geeky, conversational tone. Answer the query using official data for facts and Reddit discussions for fan opinions, without inventing details. If information is limited, say so clearly and provide a brief, factual response.

#     Instruction: {query}

#     Official Data: {vector_context}

#     Community Discussion: {reddit_context}

#     Response Guidelines:
#     - Use official data for plot, cast, ratings, or technical details.
#     - Use Reddit discussion for fan reactions or opinions, clearly labeling them as "fans say" or "Reddit users mention."
#     - Keep responses concise (2–4 sentences) unless a detailed breakdown is requested.
#     - If no relevant data is available, state: "Limited info available, but here's a general take."
#     - Maintain a fun, movie-enthusiast tone, blending facts and fan insights naturally.
#     - Avoid lists unless the query explicitly asks for them.

#     ### Response:"""
#         else:
#             return """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to provide concise, engaging, and accurate answers in a conversational, geeky tone. Use only the provided context to answer the query. If the context is insufficient, say so clearly and provide a brief, factual response based on general entertainment knowledge.

#     Instruction: {query}

#     Context: {context}

#     Response Guidelines:
#     - Answer directly using the context, avoiding invented details.
#     - Keep responses short (2–4 sentences) unless a detailed explanation is requested.
#     - Use a fun, movie-enthusiast tone, like chatting with a fellow fan.
#     - If context is missing or irrelevant, state: "Not enough context for details, but here's what I can share."
#     - Format as a natural narrative, not a list, unless the query asks for one.

#     ### Response:"""

#     # ---------------- public API ----------------------------------------
#     def generate(
#         self,
#         query: str,
#         vector_context: list[str],
#         reddit_context: list[str] | None = None,
#         max_new_tokens: int | None = None,
#         temperature: float | None = None,
#         top_k: int | None = None,
#     ) -> str:
#         """Build prompt (docs) and call LitGPT."""
#         max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
#         temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
#         top_k = top_k or rag_config["generation"]["parameters"].get("top_k", 50)

#         prompt = self._build_prompt(
#             vector_context=vector_context,
#             reddit_context=reddit_context,
#             current_query=query,
#             use_reddit=bool(reddit_context)
#         )

#         logger.debug("Prompt:\n%s", prompt)
#         if not (self.tokenizer and self.model):
#             return "[ERROR] Model/tokenizer missing."

#         # Tokenize
#         encoded = self.tokenizer.encode(prompt, device=self.device)
#         block = getattr(self.model.config, "block_size", 2048)
#         avail = block - encoded.size(0)
#         max_new_tokens = min(max_new_tokens, max(avail, 1))

#         try:
#             ids = generate(model=self.model, prompt=encoded, max_returned_tokens=encoded.size(0) + max_new_tokens, temperature=temperature, top_k=top_k, eos_id=self.tokenizer.eos_id)
#             text = self.tokenizer.decode(ids)
#             # Strip everything before ### Response: and clean up
#             response_marker = "### Response:"
#             if response_marker in text:
#                 answer = text[text.find(response_marker) + len(response_marker):].strip()
#             else:
#                 answer = text[len(prompt):].strip()
#             return answer
#         except Exception as exc:
#             logger.exception("Generation failed", exc_info=exc)
#             return "[ERROR] Generation failed."

#     def _build_prompt(self, vector_context: list[str], reddit_context: list[str] | None, current_query: str, use_reddit: bool) -> str:
#         """Build the prompt using the appropriate template."""
#         template = self.reddit_prompt_template if use_reddit else self.default_prompt_template
#         context = "\n".join(vector_context) if vector_context else "No official context available."
#         reddit_context_str = "\n".join(reddit_context) if reddit_context else "No community discussion available."
#         return template.format(
#             query=current_query,
#             vector_context=context,
#             reddit_context=reddit_context_str if use_reddit else context
#         )

#     def get_model_info(self):
#         return {
#             "litgpt_available": LITGPT_AVAILABLE,
#             "device": str(self.device),
#             "model_dir": str(MODEL_DIR),
#             "prompt_dir": str(PROMPT_DIR),
#             "default_prompt_file": DEFAULT_PROMPT_FILE,
#             "reddit_prompt_file": REDDIT_PROMPT_FILE,
#             "litgpt_dir": str(LITGPT_DIR),
#             "litgpt_dir_exists": LITGPT_DIR.exists(),
#             "model_type": type(self.model).__name__ if hasattr(self.model, "__class__") else "Unknown",
#             "load_checkpoint_available": load_checkpoint is not None,
#             "generate_function_available": generate is not None,
#             "tokenizer_loaded": self.tokenizer is not None
#         }

# if __name__ == "__main__":
#     logger.info("Starting generator initialization...")
#     try:
#         generator = Generator()
#         model_info = generator.get_model_info()
#         logger.info("Model Information:")
#         for key, value in model_info.items():
#             logger.info(f"  {key}: {value}")
#         logger.info("Generator initialized successfully!")
#     except Exception as e:
#         logger.error(f"Initialization failed: {e}")







# import torch
# import sys
# from pathlib import Path
# import logging
# import yaml

# # ----- paths ------------------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# # ----- configs ----------------------------------------------------------
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"

# with open(CONFIG_PATH, "r") as f:
#     rag_config = yaml.safe_load(f)
# with open(MODEL_CONFIG_PATH, "r") as f:
#     model_config = yaml.safe_load(f)

# # ----- imports ----------------------------------------------------------
# try:
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     AutoModelForCausalLM = AutoTokenizer = None

# # ----- logging ----------------------------------------------------------
# log_dir = BASE_DIR / rag_config["pipeline"]["logging"]["log_dir"]
# log_dir.mkdir(parents=True, exist_ok=True)
# log_file = log_dir / rag_config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(rag_config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
# )
# logger = logging.getLogger(__name__)

# # ----- constants --------------------------------------------------------
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# PROMPT_DIR = BASE_DIR / rag_config["generation"]["prompt"]["template_dir"]
# DEFAULT_PROMPT_FILE = "rag_prompt.txt"

# # -----------------------------------------------------------------------
# class Generator:
#     """Wraps Transformers model + tokenizer and builds prompts."""

#     def __init__(self):
#         logger.info("Initializing Generator with Transformers...")
#         self.device = self._get_device()
#         logger.info(f"Using device: {self.device}")
        
#         if torch.cuda.is_available():
#             torch.set_float32_matmul_precision("medium")
            
#         self.model, self.tokenizer = self._load_transformers_model()
#         self.default_prompt_template = self._load_prompt_template(DEFAULT_PROMPT_FILE)
#         logger.info("Generator ready ✔")

#     # ---------------- private helpers ----------------------------------
#     def _get_device(self):
#         """Determine the best device to use."""
#         if model_config.get("performance", {}).get("device_map"):
#             device = model_config["performance"]["device_map"]
#         else:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         return device

#     def _load_transformers_model(self):
#         """Load Mistral model using Transformers library."""
#         if not TRANSFORMERS_AVAILABLE:
#             logger.error("Transformers not available – using mock")
#             return self._mock_model(), None
            
#         try:
#             logger.info(f"Loading Mistral model: {MODEL_NAME}")
            
#             # Load tokenizer
#             tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
#             # Set pad token if not already set
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
#                 logger.info("Set pad_token to eos_token")
            
#             logger.info("Tokenizer loaded successfully")
            
#             # Load model
#             model = AutoModelForCausalLM.from_pretrained(
#                 MODEL_NAME,
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                 device_map="auto" if self.device == "cuda" else None,
#                 trust_remote_code=True
#             )
            
#             # Move to device if not using device_map="auto"
#             if self.device != "cuda":
#                 model = model.to(self.device)
            
#             model.eval()
#             logger.info("Model loaded successfully")
            
#             return model, tokenizer
            
#         except Exception as exc:
#             logger.exception("Model load failed, using mock", exc_info=exc)
#             return self._mock_model(), None

#     def _mock_model(self):
#         """Create a mock model for fallback."""
#         logger.warning("Using mock model...")
#         class Mock:
#             def generate(self, *_, **__):
#                 return "Mock response – model not loaded"
#         return Mock()

#     def _load_prompt_template(self, filename):
#         """Load prompt template from file."""
#         template_path = PROMPT_DIR / filename
#         try:
#             if not template_path.exists():
#                 default_template = self._get_default_template(filename)
#                 template_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(template_path, "w") as f:
#                     f.write(default_template)
#                 logger.info(f"Created prompt template: {template_path}")
#                 return default_template
                
#             with open(template_path, "r") as f:
#                 template = f.read()
                
#             if not template.strip():
#                 template = self._get_default_template(filename)
#                 with open(template_path, "w") as f:
#                     f.write(template)
                    
#             logger.info(f"Prompt template {filename} loaded")
#             return template
            
#         except Exception as e:
#             logger.error(f"Template {filename} loading failed: {e}")
#             return self._get_default_template(filename)

#     def _get_default_template(self, filename):
#         """Get default prompt template."""
#         return """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to provide concise, engaging, and accurate answers in a conversational, geeky tone. Use the provided context to enhance your response, but feel free to draw from your specialized entertainment knowledge to provide complete and accurate answers.

# Instruction: {query}

# Context: {vector_context}

# Response Guidelines:
# - Use the context to enhance and support your answer, combining it with your entertainment knowledge
# - Keep responses short (2–4 sentences) unless a detailed explanation is requested
# - Use a fun, movie-enthusiast tone, like chatting with a fellow fan
# - If context conflicts with known facts, prioritize accuracy over context
# - Format as a natural narrative, not a list, unless the query asks for one

# ### Response:"""

#     # ---------------- public API ----------------------------------------
#     def generate(
#         self,
#         query: str,
#         vector_context: list[str],
#         max_new_tokens: int | None = None,
#         temperature: float | None = None,
#         top_k: int | None = None,
#     ) -> str:
#         """Build prompt and generate response using Transformers."""
#         max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
#         temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
#         top_k = top_k or rag_config["generation"]["parameters"].get("top_k", 50)

#         prompt = self._build_prompt(
#             vector_context=vector_context,
#             current_query=query
#         )

#         logger.debug("Prompt:\n%s", prompt)
        
#         if not (self.tokenizer and self.model):
#             return "[ERROR] Model/tokenizer missing."

#         try:
#             # Tokenize input
#             inputs = self.tokenizer(
#                 prompt, 
#                 return_tensors="pt", 
#                 truncation=True, 
#                 max_length=2048
#             ).to(self.device)
            
#             # Generate response
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     inputs.input_ids,
#                     attention_mask=inputs.attention_mask,
#                     max_new_tokens=max_new_tokens,
#                     temperature=temperature,
#                     top_k=top_k,
#                     do_sample=True,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                 )
            
#             # Decode response
#             full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Extract only the generated part after ### Response: if present
#             response_marker = "### Response:"
#             if response_marker in full_response:
#                 answer = full_response[full_response.find(response_marker) + len(response_marker):].strip()
#             else:
#                 answer = full_response[len(prompt):].strip()
                
#             return answer
            
#         except Exception as exc:
#             logger.exception("Generation failed", exc_info=exc)
#             return "[ERROR] Generation failed."

#     def _build_prompt(self, vector_context: list[str], current_query: str) -> str:
#         """Build the prompt using the default template."""
#         template = self.default_prompt_template
#         vector_context_str = "\n".join(vector_context) if vector_context else "No official context available."
#         return template.format(
#             query=current_query,
#             vector_context=vector_context_str
#         )

#     def get_model_info(self):
#         """Get information about the loaded model."""
#         return {
#             "transformers_available": TRANSFORMERS_AVAILABLE,
#             "device": str(self.device),
#             "model_name": MODEL_NAME,
#             "prompt_dir": str(PROMPT_DIR),
#             "default_prompt_file": DEFAULT_PROMPT_FILE,
#             "model_type": type(self.model).__name__ if hasattr(self.model, "__class__") else "Unknown",
#             "tokenizer_loaded": self.tokenizer is not None,
#             "model_loaded": self.model is not None,
#             "torch_version": torch.__version__,
#             "cuda_available": torch.cuda.is_available(),
#             "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
#         }

# if __name__ == "__main__":
#     logger.info("Starting generator initialization...")
#     try:
#         generator = Generator()
#         model_info = generator.get_model_info()
#         logger.info("Model Information:")
#         for key, value in model_info.items():
#             logger.info(f"  {key}: {value}")
            
#         # Test generation
#         test_query = "What is an interesting fact about movies?"
#         test_context = ["Movies have been entertaining audiences for over a century."]
        
#         logger.info("Testing generation...")
#         response = generator.generate(
#             query=test_query,
#             vector_context=test_context,
#             max_new_tokens=100
#         )
#         logger.info(f"Test response: {response}")
#         logger.info("Generator initialized successfully!")
        
#     except Exception as e:
#         logger.error(f"Initialization failed: {e}")
#         raise











# tavily 1


# import torch
# import sys
# from pathlib import Path
# import logging
# import yaml

# # ----- paths ------------------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# # ----- configs ----------------------------------------------------------
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"

# with open(CONFIG_PATH, "r") as f:
#     rag_config = yaml.safe_load(f)
# with open(MODEL_CONFIG_PATH, "r") as f:
#     model_config = yaml.safe_load(f)

# # ----- imports ----------------------------------------------------------
# try:
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     AutoModelForCausalLM = AutoTokenizer = None

# # ----- logging ----------------------------------------------------------
# log_dir = BASE_DIR / rag_config["pipeline"]["logging"]["log_dir"]
# log_dir.mkdir(parents=True, exist_ok=True)
# log_file = log_dir / rag_config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(rag_config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
# )
# logger = logging.getLogger(__name__)

# # ----- constants --------------------------------------------------------
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# PROMPT_DIR = BASE_DIR / rag_config["generation"]["prompt"]["template_dir"]
# DEFAULT_PROMPT_FILE = "rag_prompt.txt"

# # -----------------------------------------------------------------------
# class Generator:
#     """Wraps Transformers model + tokenizer and builds prompts."""

#     def __init__(self):
#         logger.info("Initializing Generator with Transformers...")
#         self.device = self._get_device()
#         logger.info(f"Using device: {self.device}")
        
#         if torch.cuda.is_available():
#             torch.set_float32_matmul_precision("medium")
            
#         self.model, self.tokenizer = self._load_transformers_model()
#         self.default_prompt_template = self._load_prompt_template(DEFAULT_PROMPT_FILE)
#         logger.info("Generator ready ✔")

#     # ---------------- private helpers ----------------------------------
#     def _get_device(self):
#         """Determine the best device to use."""
#         if model_config.get("performance", {}).get("device_map"):
#             device = model_config["performance"]["device_map"]
#         else:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         return device

#     def _load_transformers_model(self):
#         """Load Mistral model using Transformers library."""
#         if not TRANSFORMERS_AVAILABLE:
#             logger.error("Transformers not available – using mock")
#             return self._mock_model(), None
            
#         try:
#             logger.info(f"Loading Mistral model: {MODEL_NAME}")
            
#             # Load tokenizer
#             tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
#             # Set pad token if not already set
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
#                 logger.info("Set pad_token to eos_token")
            
#             logger.info("Tokenizer loaded successfully")
            
#             # Load model
#             model = AutoModelForCausalLM.from_pretrained(
#                 MODEL_NAME,
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                 device_map="auto" if self.device == "cuda" else None,
#                 trust_remote_code=True
#             )
            
#             # Move to device if not using device_map="auto"
#             if self.device != "cuda":
#                 model = model.to(self.device)
            
#             model.eval()
#             logger.info("Model loaded successfully")
            
#             return model, tokenizer
            
#         except Exception as exc:
#             logger.exception("Model load failed, using mock", exc_info=exc)
#             return self._mock_model(), None

#     def _mock_model(self):
#         """Create a mock model for fallback."""
#         logger.warning("Using mock model...")
#         class Mock:
#             def generate(self, *_, **__):
#                 return "Mock response – model not loaded"
#         return Mock()

#     def _load_prompt_template(self, filename):
#         """Load prompt template from file."""
#         template_path = PROMPT_DIR / filename
#         try:
#             if not template_path.exists():
#                 default_template = self._get_default_template(filename)
#                 template_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(template_path, "w") as f:
#                     f.write(default_template)
#                 logger.info(f"Created prompt template: {template_path}")
#                 return default_template
                
#             with open(template_path, "r") as f:
#                 template = f.read()
                
#             if not template.strip():
#                 template = self._get_default_template(filename)
#                 with open(template_path, "w") as f:
#                     f.write(template)
                    
#             logger.info(f"Prompt template {filename} loaded")
#             return template
            
#         except Exception as e:
#             logger.error(f"Template {filename} loading failed: {e}")
#             return self._get_default_template(filename)

#     def _get_default_template(self, filename):
#         """Get default prompt template."""
#         return """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to provide concise, engaging, and accurate answers in a conversational, geeky tone. Use the provided local and web contexts to enhance your response, but feel free to draw from your specialized entertainment knowledge to provide complete and accurate answers.

# Instruction: {query}

# Local Knowledge: {local_context}

# Web Insights: {web_context}

# Response Guidelines:
# - Use the contexts to enhance and support your answer, combining them with your entertainment knowledge
# - Keep responses short (2–4 sentences) unless a detailed explanation is requested
# - Use a fun, movie-enthusiast tone, like chatting with a fellow fan
# - If contexts conflict with known facts, prioritize accuracy
# - Format as a natural narrative, not a list, unless the query asks for one

# ### Response:"""

#     # ---------------- public API ----------------------------------------
#     def generate(
#         self,
#         query: str,
#         local_context: list[str],
#         web_context: list[str],
#         max_new_tokens: int | None = None,
#         temperature: float | None = None,
#         top_k: int | None = None,
#     ) -> str:
#         """Build prompt and generate response using Transformers."""
#         max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
#         temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
#         top_k = top_k or rag_config["generation"]["parameters"].get("top_k", 50)

#         prompt = self._build_prompt(
#             local_context=local_context,
#             web_context=web_context,
#             current_query=query
#         )

#         logger.debug("Prompt:\n%s", prompt)
        
#         if not (self.tokenizer and self.model):
#             return "[ERROR] Model/tokenizer missing."

#         try:
#             # Tokenize input
#             inputs = self.tokenizer(
#                 prompt, 
#                 return_tensors="pt", 
#                 truncation=True, 
#                 max_length=2048
#             ).to(self.device)
            
#             # Generate response
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     inputs.input_ids,
#                     attention_mask=inputs.attention_mask,
#                     max_new_tokens=max_new_tokens,
#                     temperature=temperature,
#                     top_k=top_k,
#                     do_sample=True,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                 )
            
#             # Decode response
#             full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Extract only the generated part after ### Response: if present
#             response_marker = "### Response:"
#             if response_marker in full_response:
#                 answer = full_response[full_response.find(response_marker) + len(response_marker):].strip()
#             else:
#                 answer = full_response[len(prompt):].strip()
                
#             return answer
            
#         except Exception as exc:
#             logger.exception("Generation failed", exc_info=exc)
#             return "[ERROR] Generation failed."

#     def _build_prompt(self, local_context: list[str], web_context: list[str], current_query: str) -> str:
#         """Build the prompt using the default template."""
#         template = self.default_prompt_template
#         local_context_str = "\n".join(local_context) if local_context else "No local context available."
#         web_context_str = "\n".join(web_context) if web_context else "No web context available."
#         return template.format(
#             query=current_query,
#             local_context=local_context_str,
#             web_context=web_context_str
#         )

#     def get_model_info(self):
#         """Get information about the loaded model."""
#         return {
#             "transformers_available": TRANSFORMERS_AVAILABLE,
#             "device": str(self.device),
#             "model_name": MODEL_NAME,
#             "prompt_dir": str(PROMPT_DIR),
#             "default_prompt_file": DEFAULT_PROMPT_FILE,
#             "model_type": type(self.model).__name__ if hasattr(self.model, "__class__") else "Unknown",
#             "tokenizer_loaded": self.tokenizer is not None,
#             "model_loaded": self.model is not None,
#             "torch_version": torch.__version__,
#             "cuda_available": torch.cuda.is_available(),
#             "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
#         }

# if __name__ == "__main__":
#     logger.info("Starting generator initialization...")
#     try:
#         generator = Generator()
#         model_info = generator.get_model_info()
#         logger.info("Model Information:")
#         for key, value in model_info.items():
#             logger.info(f"  {key}: {value}")
            
#         # Test generation
#         test_query = "What is an interesting fact about movies?"
#         test_local_context = ["Movies have been entertaining audiences for over a century."]
#         test_web_context = ["Recent studies show that movies can improve cognitive function."]
        
#         logger.info("Testing generation...")
#         response = generator.generate(
#             query=test_query,
#             local_context=test_local_context,
#             web_context=test_web_context,
#             max_new_tokens=100
#         )
#         logger.info(f"Test response: {response}")
#         logger.info("Generator initialized successfully!")
        
#     except Exception as e:
#         logger.error(f"Initialization failed: {e}")
#         raise















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