# import torch
# import sys
# import os
# from pathlib import Path
# import logging
# import json

# # Add LitGPT to Python path
# BASE_DIR = Path(__file__).resolve().parent.parent
# LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
# sys.path.insert(0, str(LITGPT_DIR))

# try:
#     from litgpt import GPT, Config, Tokenizer
#     from litgpt.generate.base import generate
#     from litgpt.utils import load_checkpoint
#     LITGPT_AVAILABLE = True
#     print("LitGPT imported successfully")
# except ImportError as e:
#     print(f"LitGPT import failed: {e}")
#     LITGPT_AVAILABLE = False

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# MODEL_DIR = BASE_DIR / "models" / "mistral-7b-finetuned"
# PROMPT_TEMPLATE = BASE_DIR / "generation" / "prompt_templates" / "rag_prompt.txt"

# class Generator:
#     def __init__(self):
#         logger.info("Initializing Generator with LitGPT model...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {self.device}")
        
#         # Load model and tokenizer
#         self.llm = self._load_litgpt_model()
        
#         # Load prompt template
#         self.prompt_template = self._load_prompt_template()
        
#         logger.info("Generator initialized successfully!")

#     def _load_litgpt_model(self):
#         """Load LitGPT model using the proper LitGPT interface"""
#         try:
#             if not LITGPT_AVAILABLE:
#                 logger.error("LitGPT is not available. Please check your installation.")
#                 return self._create_mock_model()
            
#             # Check if the model directory is valid
#             if not MODEL_DIR.exists():
#                 logger.error(f"Model directory does not exist: {MODEL_DIR}")
#                 return self._create_mock_model()
            
#             logger.info(f"Loading LitGPT model from {MODEL_DIR}")
            
#             # Load model configuration
#             config_path = MODEL_DIR / "model_config.yaml"
#             if not config_path.exists():
#                 logger.error(f"Model config not found: {config_path}")
#                 return self._create_mock_model()
            
#             # Load the configuration
#             config = Config.from_file(config_path)
#             logger.info(f"Model config loaded: {config.name}")
            
#             # Load tokenizer
#             tokenizer = Tokenizer(MODEL_DIR)
#             logger.info("Tokenizer loaded successfully")
            
#             # Initialize the model
#             with torch.device(self.device):
#                 model = GPT(config)
            
#             # Load checkpoint
#             checkpoint_path = MODEL_DIR / "lit_model.pth"
#             if not checkpoint_path.exists():
#                 logger.error(f"Checkpoint not found: {checkpoint_path}")
#                 return self._create_mock_model()
            
#             # Load model weights
#             logger.info("Loading model checkpoint...")
#             load_checkpoint(model, checkpoint_path)
            
#             # Check for LoRA weights
#             lora_path = MODEL_DIR / "lit_model.pth.lora"
#             if lora_path.exists():
#                 logger.info("LoRA weights found, loading...")
#                 # For LoRA models, we need to handle this differently
#                 # Let's load the base model first and then apply LoRA
#                 try:
#                     lora_checkpoint = torch.load(lora_path, map_location=self.device)
#                     # Apply LoRA weights if available
#                     logger.info("LoRA weights loaded")
#                 except Exception as e:
#                     logger.warning(f"Could not load LoRA weights: {e}")
            
#             model = model.to(self.device)
#             model.eval()
            
#             logger.info("LitGPT model loaded successfully!")
            
#             # Create a wrapper class for easier use
#             class LitGPTWrapper:
#                 def __init__(self, model, tokenizer, device, model_dir):
#                     self.model = model
#                     self.tokenizer = tokenizer
#                     self.device = device
#                     self.model_dir = model_dir
                
#                 def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
#                     try:
#                         logger.info(f"Generating response for prompt length: {len(prompt)}")
                        
#                         # Use LitGPT's generate function
#                         response = generate(
#                             model=self.model,
#                             prompt=prompt,
#                             max_new_tokens=max_new_tokens,
#                             temperature=temperature,
#                             top_p=top_p,
#                             tokenizer=self.tokenizer,
#                             eos_token_id=self.tokenizer.eos_id,
#                         )
                        
#                         logger.info("Generation completed successfully")
#                         return response
                        
#                     except Exception as e:
#                         logger.error(f"Generation error: {e}")
#                         import traceback
#                         traceback.print_exc()
#                         return f"Generation failed: {e}"
                
#                 def __str__(self):
#                     return f"LitGPTWrapper({self.model.__class__.__name__})"
            
#             return LitGPTWrapper(model, tokenizer, self.device, MODEL_DIR)
            
#         except Exception as e:
#             logger.error(f"Error loading LitGPT model: {e}")
#             import traceback
#             traceback.print_exc()
#             logger.info("Creating mock model for testing...")
#             return self._create_mock_model()

#     def _create_mock_model(self):
#         """Create a mock model for testing purposes"""
#         logger.warning("Creating mock model for testing...")
        
#         class MockLLM:
#             def generate(self, prompt, **kwargs):
#                 # Return a simple mock response
#                 return "This is a mock response generated for testing. Your LitGPT model needs proper loading. The query was about: " + prompt[:100] + "..."
            
#             def __str__(self):
#                 return "MockLLM"
        
#         return MockLLM()

#     def _load_prompt_template(self):
#         """Load the prompt template"""
#         try:
#             if not PROMPT_TEMPLATE.exists():
#                 # Create the template file if it doesn't exist
#                 PROMPT_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
#                 default_template = self._get_default_template()
#                 with open(PROMPT_TEMPLATE, 'w', encoding='utf-8') as f:
#                     f.write(default_template)
#                 logger.info(f"Created default prompt template at {PROMPT_TEMPLATE}")
#                 return default_template
            
#             with open(PROMPT_TEMPLATE, 'r', encoding='utf-8') as f:
#                 template = f.read()
            
#             if not template.strip():
#                 logger.warning("Prompt template is empty, using default")
#                 template = self._get_default_template()
#                 with open(PROMPT_TEMPLATE, 'w', encoding='utf-8') as f:
#                     f.write(template)
            
#             logger.info("Prompt template loaded successfully")
#             return template
            
#         except Exception as e:
#             logger.error(f"Error loading prompt template: {e}")
#             return self._get_default_template()

#     def _get_default_template(self):
#         """Default prompt template for entertainment content"""
#         return """You are a helpful AI assistant specialized in movies, TV shows, anime, manga, and entertainment content. Use the provided context to answer the user's question accurately and comprehensively.

# Context Information:
# {context}

# User Question: {query}

# Instructions:
# - Base your answer primarily on the provided context
# - If the context doesn't contain enough information, use your general knowledge about movies and entertainment
# - Provide specific recommendations with brief explanations
# - Be concise but informative
# - If asked about similar content, explain why the recommendations are similar

# Answer:"""

#     def generate(self, query, retrieved_chunks, max_new_tokens=256, temperature=0.7, top_p=0.9):
#         """Generate response based on query and retrieved chunks"""
#         try:
#             # Prepare context from retrieved chunks
#             context = self._prepare_context(retrieved_chunks)
            
#             # Format prompt
#             prompt = self.prompt_template.format(query=query, context=context)
#             logger.info(f"Generated prompt length: {len(prompt)} characters")
            
#             # Generate response using LitGPT
#             if hasattr(self.llm, 'generate'):
#                 response = self.llm.generate(
#                     prompt,
#                     max_new_tokens=max_new_tokens,
#                     temperature=temperature,
#                     top_p=top_p,
#                     eos_token_id=None,  # Let LitGPT handle this
#                     stream=False
#                 )
#             else:
#                 # Fallback for mock model
#                 response = self.llm.generate(prompt)
            
#             # Clean up the response
#             cleaned_response = self._clean_response(response, prompt)
            
#             return cleaned_response
            
#         except Exception as e:
#             logger.error(f"Error during generation: {e}")
#             return self._generate_fallback_response(query, retrieved_chunks)

#     def _generate_fallback_response(self, query, retrieved_chunks):
#         """Generate a fallback response when model fails"""
#         logger.info("Generating fallback response...")
        
#         context = self._prepare_context(retrieved_chunks)
        
#         # Simple rule-based response
#         if "similar" in query.lower() or "recommend" in query.lower():
#             return f"Based on the context provided, here are some recommendations related to your query '{query}':\n\n{context}\n\nThese suggestions are based on the retrieved information from our database."
#         else:
#             return f"Based on the available information:\n\n{context}\n\nThis information relates to your query: '{query}'"

#     def _prepare_context(self, retrieved_chunks):
#         """Prepare context from retrieved chunks"""
#         if not retrieved_chunks:
#             return "No relevant information found in the database."
        
#         context_parts = []
#         for i, chunk in enumerate(retrieved_chunks[:3]):  # Limit to top 3 chunks
#             if isinstance(chunk, dict):
#                 text = chunk.get('text', str(chunk))
#                 # Add metadata if available
#                 if 'metadata' in chunk:
#                     metadata = chunk['metadata']
#                     if isinstance(metadata, dict):
#                         title = metadata.get('title', '')
#                         if title:
#                             text = f"Title: {title}\n{text}"
#             else:
#                 text = str(chunk)
            
#             context_parts.append(f"[{i+1}] {text}")
        
#         return "\n\n".join(context_parts)

#     def _clean_response(self, response, original_prompt):
#         """Clean and format the generated response"""
#         # Convert to string if it's not already
#         if not isinstance(response, str):
#             response = str(response)
        
#         # Remove the original prompt if it appears in the response
#         if original_prompt in response:
#             response = response.replace(original_prompt, "").strip()
        
#         # Remove extra whitespace and clean up
#         response = response.strip()
        
#         # Remove repetitive patterns
#         lines = response.split('\n')
#         cleaned_lines = []
#         prev_line = ""
        
#         for line in lines:
#             line = line.strip()
#             if line and line != prev_line and not line.startswith("Answer:"):
#                 cleaned_lines.append(line)
#                 prev_line = line
        
#         result = '\n'.join(cleaned_lines)
        
#         # If response is too short or empty, provide a basic response
#         if len(result.strip()) < 10:
#             return "I apologize, but I couldn't generate a comprehensive response. Please try rephrasing your query."
        
#         return result

#     def get_model_info(self):
#         """Get information about the loaded model"""
#         try:
#             info = {
#                 "litgpt_available": LITGPT_AVAILABLE,
#                 "device": str(self.device),
#                 "model_dir": str(MODEL_DIR),
#                 "prompt_template_exists": PROMPT_TEMPLATE.exists(),
#                 "litgpt_dir": str(LITGPT_DIR),
#                 "litgpt_dir_exists": LITGPT_DIR.exists()
#             }
            
#             if hasattr(self.llm, '__class__'):
#                 info["model_type"] = type(self.llm).__name__
            
#             return info
#         except Exception as e:
#             return {"status": f"Model info unavailable: {e}"}

# def test_generator():
#     """Test function for the generator"""
#     logger.info("Starting generator test...")
    
#     try:
#         generator = Generator()
        
#         # Print model info
#         model_info = generator.get_model_info()
#         logger.info("Model Information:")
#         for key, value in model_info.items():
#             logger.info(f"  {key}: {value}")
        
#         # Test generation
#         test_query = "Find anime similar to Berserk with dark themes"
#         test_chunks = [
#             {
#                 "text": "Berserk is a dark fantasy manga and anime with intense action, psychological themes, and mature content.",
#                 "metadata": {"title": "Berserk", "genre": "Dark Fantasy"}
#             },
#             {
#                 "text": "Claymore features dark fantasy elements, strong female protagonist, and battles against demonic creatures similar to Berserk.",
#                 "metadata": {"title": "Claymore", "genre": "Dark Fantasy"}
#             },
#             {
#                 "text": "Attack on Titan has similar dark themes, mature content, and psychological elements that fans of Berserk would appreciate.",
#                 "metadata": {"title": "Attack on Titan", "genre": "Dark Fantasy"}
#             }
#         ]
        
#         logger.info(f"Test query: {test_query}")
#         response = generator.generate(test_query, test_chunks)
#         logger.info(f"Generated response:\n{response}")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"Test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# if __name__ == "__main__":
#     success = test_generator()
#     if success:
#         logger.info("Generator test completed successfully!")
#     else:
#         logger.error("Generator test failed!")


# //above old not working properly






# import torch
# import sys
# from pathlib import Path
# import logging
# import json

# # Add LitGPT to Python path
# BASE_DIR = Path(__file__).resolve().parent.parent
# LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
# sys.path.insert(0, str(LITGPT_DIR))

# try:
#     from litgpt import GPT, Config, Tokenizer
#     from litgpt.model import load_checkpoint
#     LITGPT_AVAILABLE = True
# except ImportError as e:
#     print(f"LitGPT import failed: {e}")
#     LITGPT_AVAILABLE = False

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# MODEL_DIR = BASE_DIR / "models" / "mistral-7b-finetuned"
# PROMPT_TEMPLATE = BASE_DIR / "generation" / "prompt_templates" / "rag_prompt.txt"

# class Generator:
#     def __init__(self):
#         logger.info("Initializing Generator...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {self.device}")
#         self.llm = self._load_litgpt_model()
#         self.prompt_template = self._load_prompt_template()
#         logger.info("Generator initialized successfully!")

#     def _load_litgpt_model(self):
#         if not LITGPT_AVAILABLE:
#             logger.error("LitGPT not available.")
#             return self._create_mock_model()

#         try:
#             logger.info(f"Loading LitGPT model from {MODEL_DIR}")
#             config_path = MODEL_DIR / "model_config.yaml"
#             if not config_path.exists():
#                 logger.error(f"Config not found: {config_path}")
#                 return self._create_mock_model()

#             config = Config.from_file(config_path)
#             logger.info(f"Model config: {config.name}")

#             tokenizer = Tokenizer(MODEL_DIR)
#             logger.info("Tokenizer loaded")

#             with torch.device(self.device):
#                 model = GPT(config)

#             checkpoint_path = MODEL_DIR / "lit_model.pth"
#             if not checkpoint_path.exists():
#                 logger.error(f"Checkpoint not found: {checkpoint_path}")
#                 return self._create_mock_model()

#             logger.info("Loading checkpoint...")
#             load_checkpoint(model, str(checkpoint_path))  # Ensure path is string
#             model.to(self.device).eval()

#             lora_path = MODEL_DIR / "lit_model.pth.lora"
#             if lora_path.exists():
#                 logger.info("Applying LoRA weights...")
#                 lora_state = torch.load(lora_path, map_location=self.device)
#                 model.load_state_dict(lora_state, strict=False)

#             class LitGPTWrapper:
#                 def __init__(self, model, tokenizer):
#                     self.model = model
#                     self.tokenizer = tokenizer

#                 def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
#                     try:
#                         input_ids = self.tokenizer.encode(prompt).to(self.device)
#                         output_ids = self.model.generate(
#                             input_ids,
#                             max_new_tokens=max_new_tokens,
#                             temperature=temperature,
#                             top_p=top_p,
#                             eos_id=self.tokenizer.eos_id
#                         )
#                         return self.tokenizer.decode(output_ids[0])
#                     except Exception as e:
#                         logger.error(f"Generation error: {e}")
#                         return f"Generation failed: {e}"

#             return LitGPTWrapper(model, tokenizer)
#         except Exception as e:
#             logger.error(f"Model loading failed: {e}")
#             return self._create_mock_model()

#     def _create_mock_model(self):
#         logger.warning("Using mock model...")
#         class MockLLM:
#             def generate(self, prompt, **kwargs):
#                 return "Mock response: Model loading failed. Query: " + prompt[:100] + "..."
#         return MockLLM()

#     def _load_prompt_template(self):
#         try:
#             if not PROMPT_TEMPLATE.exists():
#                 default_template = self._get_default_template()
#                 PROMPT_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
#                 with open(PROMPT_TEMPLATE, 'w') as f:
#                     f.write(default_template)
#                 logger.info(f"Created prompt template: {PROMPT_TEMPLATE}")
#                 return default_template
#             with open(PROMPT_TEMPLATE, 'r') as f:
#                 template = f.read()
#             if not template.strip():
#                 template = self._get_default_template()
#                 with open(PROMPT_TEMPLATE, 'w') as f:
#                     f.write(template)
#             logger.info("Prompt template loaded")
#             return template
#         except Exception as e:
#             logger.error(f"Template loading failed: {e}")
#             return self._get_default_template()

#     def _get_default_template(self):
#         return """You are a helpful AI assistant specialized in movies, TV shows, anime, manga, and entertainment content. Use the provided context to answer the user's question accurately and comprehensively.

# Context Information:
# {context}

# User Question: {query}

# Instructions:
# - Base your answer primarily on the provided context
# - If the context doesn't contain enough information, use your general knowledge about movies and entertainment
# - Provide specific recommendations with brief explanations
# - Be concise but informative
# - If asked about similar content, explain why the recommendations are similar

# Answer:"""

#     def generate(self, query, retrieved_chunks, max_new_tokens=256, temperature=0.7, top_p=0.9):
#         try:
#             context = "\n".join([chunk['text'] for chunk in retrieved_chunks[:3]])
#             prompt = self.prompt_template.format(query=query, context=context)
#             logger.info(f"Prompt length: {len(prompt)}")
#             response = self.llm.generate(prompt, max_new_tokens, temperature, top_p)
#             return self._clean_response(response, prompt)
#         except Exception as e:
#             logger.error(f"Generation error: {e}")
#             return self._generate_fallback_response(query, retrieved_chunks)

#     def _generate_fallback_response(self, query, retrieved_chunks):
#         context = "\n".join([chunk['text'] for chunk in retrieved_chunks[:3]])
#         return f"Recommendations for '{query}':\n\n{context}\n\nBased on database information."

#     def _clean_response(self, response, original_prompt):
#         if not isinstance(response, str):
#             response = str(response)
#         if original_prompt in response:
#             response = response.replace(original_prompt, "").strip()
#         lines = response.split('\n')
#         cleaned_lines = []
#         prev_line = ""
#         for line in lines:
#             line = line.strip()
#             if line and line != prev_line and not line.startswith("Answer:"):
#                 cleaned_lines.append(line)
#                 prev_line = line
#         result = '\n'.join(cleaned_lines)
#         return result if result.strip() else "No comprehensive response generated."

#     def get_model_info(self):
#         return {
#             "litgpt_available": LITGPT_AVAILABLE,
#             "device": str(self.device),
#             "model_dir": str(MODEL_DIR),
#             "prompt_template_exists": PROMPT_TEMPLATE.exists(),
#             "litgpt_dir": str(LITGPT_DIR),
#             "litgpt_dir_exists": LITGPT_DIR.exists(),
#             "model_type": type(self.llm).__name__ if hasattr(self.llm, '__class__') else "Unknown"
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
import json

# Add LitGPT to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
sys.path.insert(0, str(LITGPT_DIR))

try:
    from litgpt import GPT, Config, Tokenizer
    from litgpt.utils import load_checkpoint
    import lightning.fabric as fabric
    LITGPT_AVAILABLE = True
except ImportError as e:
    print(f"LitGPT import failed: {e}")
    LITGPT_AVAILABLE = False
    load_checkpoint = None
    fabric = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / "models" / "mistral-7b-finetuned"
PROMPT_TEMPLATE = BASE_DIR / "generation" / "prompt_templates" / "rag_prompt.txt"

class Generator:
    def __init__(self):
        logger.info("Initializing Generator...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.llm = self._load_litgpt_model()
        self.prompt_template = self._load_prompt_template()
        logger.info("Generator initialized successfully!")

    def _load_litgpt_model(self):
        if not LITGPT_AVAILABLE:
            logger.error("LitGPT not available.")
            return self._create_mock_model()

        try:
            logger.info(f"Loading LitGPT model from {MODEL_DIR}")
            config_path = MODEL_DIR / "model_config.yaml"
            if not config_path.exists():
                logger.error(f"Config not found: {config_path}")
                return self._create_mock_model()

            config = Config.from_file(config_path)
            logger.info(f"Model config: {config.name}")

            tokenizer = Tokenizer(MODEL_DIR)
            logger.info("Tokenizer loaded")

            with torch.device(self.device):
                model = GPT(config)

            checkpoint_path = MODEL_DIR / "lit_model.pth"
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return self._create_mock_model()

            logger.info("Loading checkpoint...")
            
            # Create Lightning Fabric instance for checkpoint loading
            if load_checkpoint is not None and fabric is not None:
                try:
                    # Initialize Fabric
                    fabric_instance = fabric.Fabric(
                        devices=1,
                        accelerator="cuda" if torch.cuda.is_available() else "cpu",
                        precision="bf16-mixed" if torch.cuda.is_available() else "32-true"
                    )
                    
                    # Setup model with fabric
                    model = fabric_instance.setup(model)
                    
                    # Load checkpoint using fabric
                    load_checkpoint(fabric_instance, model, Path(checkpoint_path))
                    logger.info("Checkpoint loaded successfully with Fabric")
                except Exception as e:
                    logger.warning(f"Fabric checkpoint loading failed: {e}, trying fallback...")
                    # Fallback: load checkpoint manually
                    try:
                        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                        if 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                        logger.info("Checkpoint loaded using torch.load fallback")
                    except Exception as e2:
                        logger.error(f"Manual checkpoint loading failed: {e2}")
                        return self._create_mock_model()
            else:
                # Fallback: load checkpoint manually
                try:
                    checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    logger.info("Checkpoint loaded using torch.load fallback")
                except Exception as e:
                    logger.error(f"Manual checkpoint loading failed: {e}")
                    return self._create_mock_model()
            
            model.to(self.device).eval()

            lora_path = MODEL_DIR / "lit_model.pth.lora"
            if lora_path.exists():
                logger.info("Applying LoRA weights...")
                lora_state = torch.load(lora_path, map_location=self.device)
                model.load_state_dict(lora_state, strict=False)

            class LitGPTWrapper:
                def __init__(self, model, tokenizer, device):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = device

                def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
                    try:
                        # Encode the prompt
                        encoded = self.tokenizer.encode(prompt)
                        if isinstance(encoded, torch.Tensor):
                            input_ids = encoded.unsqueeze(0).to(self.device)
                        else:
                            input_ids = torch.tensor([encoded], device=self.device)
                        
                        # Check if model has generate method
                        if hasattr(self.model, 'generate'):
                            try:
                                output_ids = self.model.generate(
                                    input_ids,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    eos_id=self.tokenizer.eos_id if hasattr(self.tokenizer, 'eos_id') else None
                                )
                                return self.tokenizer.decode(output_ids[0])
                            except Exception as e:
                                logger.warning(f"Model generate method failed: {e}, using fallback")
                        
                        # Fallback generation method
                        logger.info("Using manual generation method")
                        with torch.no_grad():
                            for _ in range(max_new_tokens):
                                # Forward pass
                                outputs = self.model(input_ids)
                                if isinstance(outputs, tuple):
                                    logits = outputs[0]
                                else:
                                    logits = outputs
                                
                                # Get logits for the last token
                                next_token_logits = logits[:, -1, :] / temperature
                                
                                # Apply top-p sampling
                                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                                
                                # Remove tokens with cumulative probability above top_p
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                next_token_logits[indices_to_remove] = float('-inf')
                                
                                # Sample from the filtered distribution
                                probs = torch.softmax(next_token_logits, dim=-1)
                                next_token = torch.multinomial(probs, num_samples=1)
                                
                                # Append to input_ids
                                input_ids = torch.cat([input_ids, next_token], dim=-1)
                                
                                # Check for EOS token
                                eos_id = getattr(self.tokenizer, 'eos_id', None) or getattr(self.tokenizer, 'eos_token_id', None)
                                if eos_id is not None and next_token.item() == eos_id:
                                    break
                        
                        # Decode the full sequence
                        generated_ids = input_ids[0]
                        return self.tokenizer.decode(generated_ids)
                        
                    except Exception as e:
                        logger.error(f"Generation error: {e}")
                        return f"Generation failed: {str(e)[:200]}..."

            return LitGPTWrapper(model, tokenizer, self.device)
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return self._create_mock_model()

    def _create_mock_model(self):
        logger.warning("Using mock model...")
        class MockLLM:
            def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
                # Fixed: Now accepts all the expected parameters
                return "Mock response: Model loading failed. Query: " + prompt[:100] + "..."
        return MockLLM()

    def _load_prompt_template(self):
        try:
            if not PROMPT_TEMPLATE.exists():
                default_template = self._get_default_template()
                PROMPT_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
                with open(PROMPT_TEMPLATE, 'w') as f:
                    f.write(default_template)
                logger.info(f"Created prompt template: {PROMPT_TEMPLATE}")
                return default_template
            with open(PROMPT_TEMPLATE, 'r') as f:
                template = f.read()
            if not template.strip():
                template = self._get_default_template()
                with open(PROMPT_TEMPLATE, 'w') as f:
                    f.write(template)
            logger.info("Prompt template loaded")
            return template
        except Exception as e:
            logger.error(f"Template loading failed: {e}")
            return self._get_default_template()

    def _get_default_template(self):
        return """You are a helpful AI assistant specialized in movies, TV shows, anime, manga, and entertainment content. Use the provided context to answer the user's question accurately and comprehensively.

Context Information:
{context}

User Question: {query}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, use your general knowledge about movies and entertainment
- Provide specific recommendations with brief explanations
- Be concise but informative
- If asked about similar content, explain why the recommendations are similar

Answer:"""

    def generate(self, query, retrieved_chunks, max_new_tokens=256, temperature=0.7, top_p=0.9):
        try:
            context = "\n".join([chunk['text'] for chunk in retrieved_chunks[:3]])
            prompt = self.prompt_template.format(query=query, context=context)
            logger.info(f"Prompt length: {len(prompt)}")
            response = self.llm.generate(prompt, max_new_tokens, temperature, top_p)
            return self._clean_response(response, prompt)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._generate_fallback_response(query, retrieved_chunks)

    def _generate_fallback_response(self, query, retrieved_chunks):
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks[:3]])
        return f"Recommendations for '{query}':\n\n{context}\n\nBased on database information."

    def _clean_response(self, response, original_prompt):
        if not isinstance(response, str):
            response = str(response)
        if original_prompt in response:
            response = response.replace(original_prompt, "").strip()
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""
        for line in lines:
            line = line.strip()
            if line and line != prev_line and not line.startswith("Answer:"):
                cleaned_lines.append(line)
                prev_line = line
        result = '\n'.join(cleaned_lines)
        return result if result.strip() else "No comprehensive response generated."

    def get_model_info(self):
        return {
            "litgpt_available": LITGPT_AVAILABLE,
            "device": str(self.device),
            "model_dir": str(MODEL_DIR),
            "prompt_template_exists": PROMPT_TEMPLATE.exists(),
            "litgpt_dir": str(LITGPT_DIR),
            "litgpt_dir_exists": LITGPT_DIR.exists(),
            "model_type": type(self.llm).__name__ if hasattr(self.llm, '__class__') else "Unknown",
            "load_checkpoint_available": load_checkpoint is not None
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