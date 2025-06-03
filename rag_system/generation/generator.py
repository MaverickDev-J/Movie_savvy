# import torch
# import sys
# from pathlib import Path
# import logging
# import yaml

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
# sys.path.insert(0, str(LITGPT_DIR))

# # Load configs
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     rag_config = yaml.safe_load(f)
# with open(MODEL_CONFIG_PATH, "r") as f:
#     model_config = yaml.safe_load(f)

# try:
#     from litgpt import GPT, Config, Tokenizer
#     from litgpt.utils import load_checkpoint
#     import lightning.fabric as fabric
#     LITGPT_AVAILABLE = True
# except ImportError as e:
#     print(f"LitGPT import failed: {e}")
#     LITGPT_AVAILABLE = False
#     load_checkpoint = None
#     fabric = None

# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(BASE_DIR / rag_config["pipeline"]["logging"]["log_dir"] / rag_config["pipeline"]["logging"]["log_file"]),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# MODEL_DIR = BASE_DIR / model_config["fine_tuned_model"]["path"]
# PROMPT_TEMPLATE = BASE_DIR / rag_config["generation"]["prompt"]["template_file"]

# class Generator:
#     def __init__(self):
#         logger.info("Initializing Generator...")
#         self.device = model_config["performance"]["device_map"] if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {self.device}")
#         # Optimize for Tensor Cores
#         if torch.cuda.is_available():
#             torch.set_float32_matmul_precision("medium")
#         self.llm = self._load_litgpt_model()
#         self.prompt_template = self._load_prompt_template()
#         logger.info("Generator initialized successfully!")

#     def _load_litgpt_model(self):
#         if not LITGPT_AVAILABLE:
#             logger.error("LitGPT not available.")
#             return self._create_mock_model()

#         try:
#             logger.info(f"Loading LitGPT model from {MODEL_DIR}")
#             config_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["model_config"]
#             if not config_path.exists():
#                 logger.error(f"Config not found: {config_path}")
#                 return self._create_mock_model()

#             config = Config.from_file(config_path)
#             logger.info(f"Model config: {config.name}")

#             tokenizer = Tokenizer(MODEL_DIR)
#             logger.info("Tokenizer loaded")

#             with torch.device(self.device):
#                 model = GPT(config)

#             checkpoint_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["checkpoint"]
#             if not checkpoint_path.exists():
#                 logger.error(f"Checkpoint not found: {checkpoint_path}")
#                 return self._create_mock_model()

#             logger.info("Loading checkpoint...")
#             if load_checkpoint is not None and fabric is not None:
#                 try:
#                     fabric_instance = fabric.Fabric(
#                         devices=model_config["loading"]["fabric_devices"],
#                         accelerator=model_config["loading"]["fabric_accelerator"],
#                         precision=model_config["training"]["precision"]
#                     )
#                     model = fabric_instance.setup(model)
#                     load_checkpoint(fabric_instance, model, Path(checkpoint_path))
#                     logger.info("Checkpoint loaded successfully with Fabric")
#                 except Exception as e:
#                     logger.warning(f"Fabric checkpoint loading failed: {e}, trying fallback...")
#                     try:
#                         checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
#                         model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
#                         logger.info("Checkpoint loaded using torch.load fallback")
#                     except Exception as e2:
#                         logger.error(f"Manual checkpoint loading failed: {e2}")
#                         return self._create_mock_model()
#             else:
#                 try:
#                     checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
#                     model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
#                     logger.info("Checkpoint loaded using torch.load fallback")
#                 except Exception as e:
#                     logger.error(f"Manual checkpoint loading failed: {e}")
#                     return self._create_mock_model()

#             model.to(self.device).eval()

#             lora_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["lora_weights"]
#             if lora_path.exists():
#                 logger.info("Applying LoRA weights...")
#                 try:
#                     lora_state = torch.load(lora_path, map_location=self.device)
#                     model.load_state_dict(lora_state, strict=False)
#                     logger.info("LoRA weights applied successfully")
#                 except Exception as e:
#                     logger.warning(f"Failed to apply LoRA weights: {e}")

#             class LitGPTWrapper:
#                 def __init__(self, model, tokenizer, device):
#                     self.model = model
#                     self.tokenizer = tokenizer
#                     self.device = device

#                 def generate(self, prompt, max_new_tokens, temperature, top_p):
#                     try:
#                         encoded = self.tokenizer.encode(prompt)
#                         input_ids = encoded.unsqueeze(0).to(self.device) if isinstance(encoded, torch.Tensor) else torch.tensor([encoded], device=self.device)

#                         if hasattr(self.model, "generate"):
#                             try:
#                                 output_ids = self.model.generate(
#                                     input_ids,
#                                     max_new_tokens=max_new_tokens,
#                                     temperature=temperature,
#                                     top_p=top_p,
#                                     eos_id=getattr(self.tokenizer, "eos_id", None)
#                                 )
#                                 return self.tokenizer.decode(output_ids[0])
#                             except Exception as e:
#                                 logger.warning(f"Model generate method failed: {e}, using fallback")

#                         logger.info("Using manual generation method")
#                         with torch.no_grad():
#                             for _ in range(max_new_tokens):
#                                 outputs = self.model(input_ids)
#                                 logits = outputs[0] if isinstance(outputs, tuple) else outputs
#                                 next_token_logits = logits[:, -1, :] / temperature
#                                 sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
#                                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
#                                 sorted_indices_to_remove = cumulative_probs > top_p
#                                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#                                 sorted_indices_to_remove[..., 0] = 0
#                                 indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#                                 next_token_logits[indices_to_remove] = float("-inf")
#                                 probs = torch.softmax(next_token_logits, dim=-1)
#                                 next_token = torch.multinomial(probs, num_samples=1)
#                                 input_ids = torch.cat([input_ids, next_token], dim=-1)
#                                 eos_id = getattr(self.tokenizer, "eos_id", getattr(self.tokenizer, "eos_token_id", None))
#                                 if eos_id is not None and next_token.item() == eos_id:
#                                     break
#                         return self.tokenizer.decode(input_ids[0])
#                     except Exception as e:
#                         logger.error(f"Generation error: {e}")
#                         return f"Generation failed: {str(e)[:200]}..."

#             return LitGPTWrapper(model, tokenizer, self.device)
#         except Exception as e:
#             logger.error(f"Model loading failed: {e}")
#             return self._create_mock_model()

#     def _create_mock_model(self):
#         logger.warning("Using mock model...")
#         class MockLLM:
#             def generate(self, prompt, max_new_tokens, temperature, top_p):
#                 return f"Mock response: Model loading failed. Query: {prompt[:100]}..."
#         return MockLLM()

#     def _load_prompt_template(self):
#         try:
#             if not PROMPT_TEMPLATE.exists():
#                 default_template = self._get_default_template()
#                 PROMPT_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
#                 with open(PROMPT_TEMPLATE, "w") as f:
#                     f.write(default_template)
#                 logger.info(f"Created prompt template: {PROMPT_TEMPLATE}")
#                 return default_template
#             with open(PROMPT_TEMPLATE, "r") as f:
#                 template = f.read()
#             if not template.strip():
#                 template = self._get_default_template()
#                 with open(PROMPT_TEMPLATE, "w") as f:
#                     f.write(template)
#             logger.info("Prompt template loaded")
#             return template
#         except Exception as e:
#             logger.error(f"Template loading failed: {e}")
#             return self._get_default_template()

#     def _get_default_template(self):
#         return """You are a movie-savvy AI assistant specializing in entertainment content (movies, anime, manga, etc.). Answer the user's query strictly based on the provided context, avoiding any invented details. If the context is insufficient, state so clearly and provide a concise response based on available information.

# Context Information:
# {context}

# User Question: {query}

# Instructions:
# - Use only the provided context for specific details.
# - Be concise, informative, and maintain a fun, movie-geek tone.
# - For recommendations or comparisons, explain relevance briefly.
# - If context lacks details, say "Insufficient context for a detailed response" and provide a general answer.

# Answer:"""

#     def generate(self, query, retrieved_chunks, max_new_tokens=None, temperature=None, top_p=None):
#         try:
#             max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
#             temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
#             top_p = top_p or rag_config["generation"]["parameters"]["top_p"]
#             context = "\n".join([chunk["text"] for chunk in retrieved_chunks[:rag_config["retrieval"]["top_k"]]])
#             prompt = self.prompt_template.format(query=query, context=context)
#             logger.info(f"Prompt length: {len(prompt)}")
#             response = self.llm.generate(prompt, max_new_tokens, temperature, top_p)
#             return self._clean_response(response, prompt)
#         except Exception as e:
#             logger.error(f"Generation error: {e}")
#             return self._generate_fallback_response(query, retrieved_chunks)

#     def _generate_fallback_response(self, query, retrieved_chunks):
#         context = "\n".join([chunk["text"] for chunk in retrieved_chunks[:rag_config["retrieval"]["top_k"]]])
#         return f"Insufficient context for '{query}'. Based on available data:\n\n{context}"

#     def _clean_response(self, response, original_prompt):
#         if not isinstance(response, str):
#             response = str(response)
#         if original_prompt in response:
#             response = response.replace(original_prompt, "").strip()
#         lines = response.split("\n")
#         cleaned_lines = [line.strip() for line in lines if line.strip() and not line.startswith("Answer:")]
#         result = "\n".join(cleaned_lines)
#         return result if result.strip() else "No comprehensive response generated."

#     def get_model_info(self):
#         return {
#             "litgpt_available": LITGPT_AVAILABLE,
#             "device": str(self.device),
#             "model_dir": str(MODEL_DIR),
#             "prompt_template_exists": PROMPT_TEMPLATE.exists(),
#             "litgpt_dir": str(LITGPT_DIR),
#             "litgpt_dir_exists": LITGPT_DIR.exists(),
#             "model_type": type(self.llm).__name__ if hasattr(self.llm, "__class__") else "Unknown",
#             "load_checkpoint_available": load_checkpoint is not None
#         }

# def test_generator():
#     logger.info("Starting generator test...")
#     try:
#         generator = Generator()
#         model_info = generator.get_model_info()
#         logger.info("Model Information:")
#         for key, value in model_info.items():
#             logger.info(f"  {key}: {value}")
#         test_query = "Find anime similar to Berserk with dark themes"
#         test_chunks = [
#             {"text": "Berserk is a dark fantasy manga and anime with intense action, psychological themes, and mature content."},
#             {"text": "Claymore features dark fantasy elements, strong female protagonist, and battles against demonic creatures similar to Berserk."},
#             {"text": "Attack on Titan has similar dark themes, mature content, and psychological elements that fans of Berserk would appreciate."}
#         ]
#         logger.info(f"Test query: {test_query}")
#         response = generator.generate(test_query, test_chunks)
#         logger.info(f"Generated response:\n{response}")
#         return True
#     except Exception as e:
#         logger.error(f"Test failed: {e}")
#         return False

# if __name__ == "__main__":
#     success = test_generator()
#     logger.info("Generator test completed successfully!" if success else "Generator test failed!")








import torch
import sys
from pathlib import Path
import logging
import yaml

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
sys.path.insert(0, str(LITGPT_DIR))

# Load configs
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
MODEL_CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"
with open(CONFIG_PATH, "r") as f:
    rag_config = yaml.safe_load(f)
with open(MODEL_CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)

try:
    from litgpt import GPT, Config, Tokenizer
    from litgpt.utils import load_checkpoint
    from litgpt.generate.base import generate  # Import generate function
    import lightning.fabric as fabric
    LITGPT_AVAILABLE = True
except ImportError as e:
    print(f"LitGPT import failed: {e}")
    LITGPT_AVAILABLE = False
    load_checkpoint = None
    fabric = None
    generate = None

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / rag_config["pipeline"]["logging"]["log_dir"] / rag_config["pipeline"]["logging"]["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / model_config["fine_tuned_model"]["path"]
PROMPT_TEMPLATE = BASE_DIR / rag_config["generation"]["prompt"]["template_file"]

class Generator:
    def __init__(self):
        logger.info("Initializing Generator...")
        self.device = model_config["performance"]["device_map"] if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        # Optimize for Tensor Cores
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
        self.model, self.tokenizer = self._load_litgpt_model()
        self.prompt_template = self._load_prompt_template()
        logger.info("Generator initialized successfully!")

    def _load_litgpt_model(self):
        if not LITGPT_AVAILABLE:
            logger.error("LitGPT not available.")
            return self._create_mock_model(), None

        try:
            logger.info(f"Loading LitGPT model from {MODEL_DIR}")
            config_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["model_config"]
            if not config_path.exists():
                logger.error(f"Config not found: {config_path}")
                return self._create_mock_model(), None

            config = Config.from_file(config_path)
            logger.info(f"Model config: {config.name}")

            tokenizer = Tokenizer(MODEL_DIR)
            logger.info("Tokenizer loaded")

            # Initialize model
            with torch.device(self.device):
                model = GPT(config)

            checkpoint_path = MODEL_DIR / model_config["fine_tuned_model"]["files"]["checkpoint"]
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return self._create_mock_model(), None

            logger.info("Loading checkpoint...")
            if load_checkpoint is not None and fabric is not None:
                try:
                    fabric_instance = fabric.Fabric(
                        devices=model_config["loading"]["fabric_devices"],
                        accelerator=model_config["loading"]["fabric_accelerator"],
                        precision=model_config["training"]["precision"]
                    )
                    model = fabric_instance.setup(model)
                    load_checkpoint(fabric_instance, model, Path(checkpoint_path))
                    logger.info("Checkpoint loaded successfully with Fabric")
                except Exception as e:
                    logger.warning(f"Fabric checkpoint loading failed: {e}, trying fallback...")
                    try:
                        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                        model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
                        logger.info("Checkpoint loaded using torch.load fallback")
                    except Exception as e2:
                        logger.error(f"Manual checkpoint loading failed: {e2}")
                        return self._create_mock_model(), None
            else:
                try:
                    checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
                    logger.info("Checkpoint loaded using torch.load fallback")
                except Exception as e:
                    logger.error(f"Manual checkpoint loading failed: {e}")
                    return self._create_mock_model(), None

            model.to(self.device).eval()

            # FIXED: Initialize KV cache after model setup - this is the key fix!
            model.set_kv_cache(batch_size=1, device=self.device)
            logger.info("KV cache initialized")

            # Apply LoRA weights if available
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
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return self._create_mock_model(), None

    def _create_mock_model(self):
        logger.warning("Using mock model...")
        class MockLLM:
            def generate(self, prompt, max_new_tokens, temperature, top_k):
                return f"Mock response: Model loading failed. Query: {prompt[:100]}..."
        return MockLLM()

    def _load_prompt_template(self):
        try:
            if not PROMPT_TEMPLATE.exists():
                default_template = self._get_default_template()
                PROMPT_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
                with open(PROMPT_TEMPLATE, "w") as f:
                    f.write(default_template)
                logger.info(f"Created prompt template: {PROMPT_TEMPLATE}")
                return default_template
            with open(PROMPT_TEMPLATE, "r") as f:
                template = f.read()
            if not template.strip():
                template = self._get_default_template()
                with open(PROMPT_TEMPLATE, "w") as f:
                    f.write(template)
            logger.info("Prompt template loaded")
            return template
        except Exception as e:
            logger.error(f"Template loading failed: {e}")
            return self._get_default_template()

    def _get_default_template(self):
        return """You are a movie-savvy AI assistant specializing in entertainment content (movies, anime, manga, etc.). Answer the user's query strictly based on the provided context, avoiding any invented details. If the context is insufficient, state so clearly and provide a concise response based on available information.

Context Information:
{context}

User Question: {query}

Instructions:
- Use only the provided context for specific details.
- Be concise, informative, and maintain a fun, movie-geek tone.
- For recommendations or comparisons, explain relevance briefly.
- If context lacks details, say "Insufficient context for a detailed response" and provide a general answer.

Answer:"""

    # def generate(self, query, retrieved_chunks, max_new_tokens=None, temperature=None, top_k=None):
    #     try:
    #         # Get generation parameters
    #         max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
    #         temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
    #         top_k = top_k or rag_config["generation"]["parameters"].get("top_k", 50)  # Default to 50 if not specified
            
    #         # Prepare context and prompt
    #         context = "\n".join([chunk["text"] for chunk in retrieved_chunks[:rag_config["retrieval"]["top_k"]]])
    #         prompt = self.prompt_template.format(query=query, context=context)
    #         logger.info(f"Prompt length: {len(prompt)}")
            
    #         # Check if we have a valid model and tokenizer
    #         if self.tokenizer is None or self.model is None:
    #             logger.error("Model or tokenizer not loaded properly")
    #             return self._generate_fallback_response(query, retrieved_chunks)
            
    #         # Use LitGPT's native generate function
    #         if generate is not None and LITGPT_AVAILABLE:
    #             try:
    #                 logger.info("Using LitGPT native generate function")
                    
    #                 # Tokenize the prompt
    #                 encoded_prompt = self.tokenizer.encode(prompt, device=self.device)
    #                 logger.info(f"Encoded prompt shape: {encoded_prompt.shape if hasattr(encoded_prompt, 'shape') else 'N/A'}")
                    
    #                 # Use LitGPT's generate function with correct parameter names
    #                 with torch.no_grad():
    #                     response_ids = generate(
    #                         model=self.model,
    #                         prompt=encoded_prompt,
    #                         max_returned_tokens=max_new_tokens,  # FIXED: Use correct parameter name
    #                         temperature=temperature,
    #                         top_k=top_k,
    #                         eos_id=self.tokenizer.eos_id  # FIXED: Add eos_id for proper stopping
    #                     )
                    
    #                 # Decode the response back to text
    #                 response = self.tokenizer.decode(response_ids)
    #                 logger.info("Successfully used LitGPT native generate function")
    #                 return self._clean_response(response, prompt)
                    
    #             except Exception as e:
    #                 logger.warning(f"LitGPT native generate failed: {e}, falling back to manual generation")
            
    #         # Fallback to manual generation if native generate fails
    #         logger.info("Using manual generation method")
    #         response = self._manual_generate(prompt, max_new_tokens, temperature, top_k)
    #         return self._clean_response(response, prompt)
            
    #     except Exception as e:
    #         logger.error(f"Generation error: {e}")
    #         return self._generate_fallback_response(query, retrieved_chunks)


    def generate(self, query, retrieved_chunks, max_new_tokens=None, temperature=None, top_k=None):
        try:
            # Get generation parameters
            max_new_tokens = max_new_tokens or rag_config["generation"]["parameters"]["max_new_tokens"]
            temperature = temperature or rag_config["generation"]["parameters"]["temperature"]
            top_k = top_k or rag_config["generation"]["parameters"].get("top_k", 50)
            
            # Prepare context and prompt
            context = "\n".join([chunk["text"] for chunk in retrieved_chunks[:rag_config["retrieval"]["top_k"]]])
            prompt = self.prompt_template.format(query=query, context=context)
            logger.info(f"Prompt length: {len(prompt)}")
            
            # Check if we have a valid model and tokenizer
            if self.tokenizer is None or self.model is None:
                logger.error("Model or tokenizer not loaded properly")
                return self._generate_fallback_response(query, retrieved_chunks)
            
            # Use LitGPT's native generate function
            if generate is not None and LITGPT_AVAILABLE:
                try:
                    logger.info("Using LitGPT native generate function")
                    
                    # Tokenize the prompt
                    encoded_prompt = self.tokenizer.encode(prompt, device=self.device)
                    logger.info(f"Encoded prompt shape: {encoded_prompt.shape if hasattr(encoded_prompt, 'shape') else 'N/A'}")
                    
                    # 🔥 CRITICAL FIX: Clamp max_new_tokens to available context
                    block_size = getattr(self.model.config, "block_size", 2048)
                    prompt_length = encoded_prompt.size(0) if hasattr(encoded_prompt, 'size') else len(encoded_prompt)
                    available_tokens = block_size - prompt_length
                    
                    if available_tokens <= 0:
                        logger.warning(f"Prompt too long ({prompt_length} tokens) for model context ({block_size}). Truncating context.")
                        # Truncate context and recreate prompt
                        truncated_context = context[:len(context)//2]  # Simple truncation
                        prompt = self.prompt_template.format(query=query, context=truncated_context)
                        encoded_prompt = self.tokenizer.encode(prompt, device=self.device)
                        prompt_length = encoded_prompt.size(0) if hasattr(encoded_prompt, 'size') else len(encoded_prompt)
                        available_tokens = block_size - prompt_length
                    
                    if max_new_tokens > available_tokens:
                        logger.warning(f"max_new_tokens {max_new_tokens} > available context {available_tokens}; clamping to {available_tokens}")
                        max_new_tokens = max(available_tokens, 1)  # Ensure at least 1 token
                    
                    # Use LitGPT's generate function with correct parameter names
                    with torch.no_grad():
                        response_ids = generate(
                            model=self.model,
                            prompt=encoded_prompt,
                            max_returned_tokens=prompt_length+max_new_tokens,  # FIXED: Use correct parameter name
                            temperature=temperature,
                            top_k=top_k,
                            eos_id=self.tokenizer.eos_id
                        )
                    
                    # Decode the response back to text
                    response = self.tokenizer.decode(response_ids)
                    logger.info("Successfully used LitGPT native generate function")
                    return self._clean_response(response, prompt)
                    
                except Exception as e:
                    logger.exception(f"LitGPT native generate failed: {e}, falling back to manual generation")
            
            # Fallback to manual generation if native generate fails
            logger.info("Using manual generation method")
            response = self._manual_generate(prompt, max_new_tokens, temperature, top_k)
            return self._clean_response(response, prompt)
            
        except Exception as e:
            logger.exception(f"Generation error: {e}")  # Use exception() for full traceback
            return self._generate_fallback_response(query, retrieved_chunks)
        





    def _manual_generate(self, prompt, max_new_tokens, temperature, top_k):
        """Manual generation as fallback"""
        try:
            encoded = self.tokenizer.encode(prompt)
            input_ids = encoded.unsqueeze(0).to(self.device) if isinstance(encoded, torch.Tensor) else torch.tensor([encoded], device=self.device)

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(input_ids)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Top-k sampling (instead of top-p)
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        # Set all non-top-k logits to -inf
                        next_token_logits.fill_(float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    
                    # Check for EOS token
                    eos_id = getattr(self.tokenizer, "eos_id", getattr(self.tokenizer, "eos_token_id", None))
                    if eos_id is not None and next_token.item() == eos_id:
                        break
                        
            return self.tokenizer.decode(input_ids[0])
            
        except Exception as e:
            logger.error(f"Manual generation error: {e}")
            return f"Generation failed: {str(e)[:200]}..."

    def _generate_fallback_response(self, query, retrieved_chunks):
        context = "\n".join([chunk["text"] for chunk in retrieved_chunks[:rag_config["retrieval"]["top_k"]]])
        return f"Insufficient context for '{query}'. Based on available data:\n\n{context}"

    def _clean_response(self, response, original_prompt):
        if not isinstance(response, str):
            response = str(response)
        
        # Remove the original prompt if it appears in the response
        if original_prompt in response:
            response = response.replace(original_prompt, "").strip()
        
        # Clean up the response
        lines = response.split("\n")
        cleaned_lines = [line.strip() for line in lines if line.strip() and not line.startswith("Answer:")]
        result = "\n".join(cleaned_lines)
        
        return result if result.strip() else "No comprehensive response generated."

    def get_model_info(self):
        return {
            "litgpt_available": LITGPT_AVAILABLE,
            "device": str(self.device),
            "model_dir": str(MODEL_DIR),
            "prompt_template_exists": PROMPT_TEMPLATE.exists(),
            "litgpt_dir": str(LITGPT_DIR),
            "litgpt_dir_exists": LITGPT_DIR.exists(),
            "model_type": type(self.model).__name__ if hasattr(self.model, "__class__") else "Unknown",
            "load_checkpoint_available": load_checkpoint is not None,
            "generate_function_available": generate is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }

def test_generator():
    logger.info("Starting generator test...")
    try:
        generator = Generator()
        model_info = generator.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        test_query = "Find anime similar to Berserk with dark themes"
        test_chunks = [
            {"text": "Berserk is a dark fantasy manga and anime with intense action, psychological themes, and mature content."},
            {"text": "Claymore features dark fantasy elements, strong female protagonist, and battles against demonic creatures similar to Berserk."},
            {"text": "Attack on Titan has similar dark themes, mature content, and psychological elements that fans of Berserk would appreciate."}
        ]
        
        logger.info(f"Test query: {test_query}")
        response = generator.generate(test_query, test_chunks)
        logger.info(f"Generated response:\n{response}")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_generator()
    logger.info("Generator test completed successfully!" if success else "Generator test failed!")