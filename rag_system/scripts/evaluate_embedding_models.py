# import os
# import json
# import yaml
# import logging
# import shutil
# import subprocess
# import time
# import numpy as np
# from pathlib import Path
# import mlflow
# import psutil
# import torch
# from contextlib import contextmanager
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('rag_system/output/logs/evaluate_embedding_models.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Define paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# TEST_QUERIES_PATH = BASE_DIR / "data" / "test_queries.json"
# OUTPUT_DIR = BASE_DIR / "output" / "evaluation"
# EMBEDDINGS_DIR = BASE_DIR / "data" / "processed" / "embeddings"
# INDEX_DIR = BASE_DIR / "retrieval" / "index"

# # Ensure output directory exists
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Embedding models to evaluate with their dimensions
# EMBEDDING_MODELS = {
#     "intfloat/e5-base-v2": 768,
#     "BAAI/bge-base-en-v1.5": 768,
#     "bge-large-en-v1.5": 1024,
#     "e5-large-v2": 1024
# }

# @contextmanager
# def track_memory():
#     """Track peak RAM usage during a block of code."""
#     process = psutil.Process()
#     peak_memory = 0
#     try:
#         yield
#         peak_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
#     finally:
#         logger.info(f"Peak RAM usage: {peak_memory:.2f} MB")

# def load_config():
#     """Load rag_config.yaml."""
#     try:
#         with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
#             return yaml.safe_load(f)
#     except Exception as e:
#         logger.error(f"Failed to load config: {e}")
#         raise

# def save_config(config):
#     """Save config to rag_config.yaml."""
#     try:
#         with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
#             yaml.safe_dump(config, f)
#     except Exception as e:
#         logger.error(f"Failed to save config: {e}")
#         raise

# def load_test_queries():
#     """Load test queries from JSON file."""
#     try:
#         with open(TEST_QUERIES_PATH, 'r', encoding='utf-8') as f:
#             queries = json.load(f)
#         logger.info(f"Loaded {len(queries)} test queries")
#         return queries
#     except Exception as e:
#         logger.error(f"Failed to load test queries: {e}")
#         raise

# def run_script(script_name, args=None):
#     """Run a Python script and measure execution time."""
#     cmd = ["python", f"rag_system/scripts/{script_name}"]
#     if args:
#         cmd.extend(args)
#     start_time = time.time()
#     with track_memory():
#         result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#     duration = time.time() - start_time
#     logger.info(f"Ran {script_name} in {duration:.2f} seconds")
#     logger.debug(f"Output: {result.stdout}")
#     return duration

# def run_retrieval(query, top_k):
#     """Run retrieval and measure time and similarity scores."""
#     from rag_system.retrieval.retriever import retrieve
#     import asyncio
#     start_time = time.time()
#     with track_memory():
#         results = asyncio.run(retrieve(query, top_k))
#     duration = time.time() - start_time
#     similarities = [res['similarity'] for res in results]
#     mean_similarity = float(np.mean(similarities)) if similarities else 0.0
#     logger.info(f"Retrieval for query '{query[:30]}...': {len(results)} chunks, mean similarity: {mean_similarity:.4f}")
#     return results, duration, mean_similarity

# def run_generation(query, top_k):
#     """Run RAG pipeline for a query and measure generation time."""
#     cmd = ["python", "rag_system/scripts/run_rag.py", "--query", query, "--top_k", str(top_k)]
#     start_time = time.time()
#     with track_memory():
#         result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#     duration = time.time() - start_time
#     output_file = OUTPUT_DIR / f"result_{''.join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')}.json"
#     try:
#         with open(output_file, 'r', encoding='utf-8') as f:
#             result_data = json.load(f)
#         response = result_data['response']
#     except Exception as e:
#         logger.error(f"Failed to read RAG output: {e}")
#         response = "Error in generation"
#     logger.info(f"Generation for query '{query[:30]}...': {duration:.2f} seconds")
#     return response, duration

# def main():
#     """Run MLflow experiment to evaluate embedding models."""
#     # Set up MLflow
#     mlflow.set_experiment("EmbeddingModelComparison")
    
#     # Load original config and queries
#     original_config = load_config()
#     test_queries = load_test_queries()
#     top_k = original_config['retrieval']['top_k']
    
#     # Back up original config
#     config_backup = CONFIG_PATH.with_suffix('.yaml.bak')
#     shutil.copy(CONFIG_PATH, config_backup)
#     logger.info(f"Backed up original config to {config_backup}")
    
#     try:
#         for model_name, emb_dim in EMBEDDING_MODELS.items():
#             logger.info(f"Evaluating embedding model: {model_name}")
#             with mlflow.start_run(run_name=model_name):
#                 # Update config with model-specific settings
#                 config = load_config()
#                 config['retrieval']['embedding_model'] = model_name
#                 config['retrieval']['embedding_dimension'] = emb_dim
#                 save_config(config)
                
#                 # Log parameters
#                 mlflow.log_param("embedding_model", model_name)
#                 mlflow.log_param("embedding_dimension", emb_dim)
#                 mlflow.log_param("top_k", top_k)
#                 mlflow.log_param("batch_size", config['pipeline']['batch_size'])
                
#                 # Clear previous embeddings and index
#                 if EMBEDDINGS_DIR.exists():
#                     shutil.rmtree(EMBEDDINGS_DIR)
#                     EMBEDDINGS_DIR.mkdir(parents=True)
#                 if INDEX_DIR.exists():
#                     shutil.rmtree(INDEX_DIR)
#                     INDEX_DIR.mkdir(parents=True)
                
#                 # Run embedding generation
#                 logger.info("Running embedder.py...")
#                 embedding_time = run_script("embedder.py")
#                 mlflow.log_metric("embedding_time", embedding_time)
                
#                 # Run index building
#                 logger.info("Running build_index.py...")
#                 index_time = run_script("build_index.py")
#                 mlflow.log_metric("index_build_time", index_time)
                
#                 # Evaluate queries
#                 total_retrieval_time = 0
#                 total_generation_time = 0
#                 mean_similarities = []
#                 sample_responses = {}
                
#                 for query in tqdm(test_queries, desc=f"Evaluating queries for {model_name}"):
#                     # Run retrieval
#                     results, retrieval_time, mean_similarity = run_retrieval(query, top_k)
#                     total_retrieval_time += retrieval_time
#                     mean_similarities.append(mean_similarity)
                    
#                     # Run generation
#                     response, generation_time = run_generation(query, top_k)
#                     total_generation_time += generation_time
#                     sample_responses[query] = response
                    
#                     # Log query-specific metrics
#                     mlflow.log_metric(f"retrieval_time_{query[:20]}", retrieval_time)
#                     mlflow.log_metric(f"generation_time_{query[:20]}", generation_time)
#                     mlflow.log_metric(f"similarity_{query[:20]}", mean_similarity)
                
#                 # Log aggregate metrics
#                 avg_retrieval_time = total_retrieval_time / len(test_queries)
#                 avg_generation_time = total_generation_time / len(test_queries)
#                 avg_similarity = float(np.mean(mean_similarities))
#                 mlflow.log_metric("avg_retrieval_time", avg_retrieval_time)
#                 mlflow.log_metric("avg_generation_time", avg_generation_time)
#                 mlflow.log_metric("avg_similarity_score", avg_similarity)
                
#                 # Log sample responses as artifact
#                 responses_file = OUTPUT_DIR / f"responses_{model_name.replace('/', '_')}.json"
#                 with open(responses_file, 'w', encoding='utf-8') as f:
#                     json.dump(sample_responses, f, indent=2)
#                 mlflow.log_artifact(str(responses_file))
                
#                 # Log config and logs
#                 mlflow.log_artifact(str(CONFIG_PATH))
#                 mlflow.log_artifact("rag_system/output/logs/evaluate_embedding_models.log")
                
#                 logger.info(f"Completed evaluation for {model_name}")
#                 logger.info(f"Average similarity: {avg_similarity:.4f}")
#                 logger.info(f"Average retrieval time: {avg_retrieval_time:.2f} seconds")
#                 logger.info(f"Average generation time: {avg_generation_time:.2f} seconds")
    
#     finally:
#         # Restore original config
#         shutil.move(config_backup, CONFIG_PATH)
#         logger.info("Restored original config")
    
#     # Generate summary report
#     summary = {
#         "models": [],
#         "avg_similarity_scores": [],
#         "avg_retrieval_times": [],
#         "avg_generation_times": [],
#         "embedding_times": [],
#         "index_build_times": []
#     }
#     for model_name in EMBEDDING_MODELS:
#         with mlflow.start_run(run_name=model_name, nested=True):
#             run = mlflow.active_run()
#             if run:
#                 metrics = mlflow.get_run(run.info.run_id).data.metrics
#                 summary["models"].append(model_name)
#                 summary["avg_similarity_scores"].append(metrics.get("avg_similarity_score", 0))
#                 summary["avg_retrieval_times"].append(metrics.get("avg_retrieval_time", 0))
#                 summary["avg_generation_times"].append(metrics.get("avg_generation_time", 0))
#                 summary["embedding_times"].append(metrics.get("embedding_time", 0))
#                 summary["index_build_times"].append(metrics.get("index_build_time", 0))
    
#     summary_file = OUTPUT_DIR / "evaluation_summary.json"
#     with open(summary_file, 'w', encoding='utf-8') as f:
#         json.dump(summary, f, indent=2)
#     mlflow.log_artifact(str(summary_file))
#     logger.info(f"Saved evaluation summary to {summary_file}")

# if __name__ == "__main__":
#     main()



# import os
# import json
# import yaml
# import logging
# import shutil
# import subprocess
# import time
# import numpy as np
# from pathlib import Path
# import mlflow
# import psutil
# import torch
# from contextlib import contextmanager
# from tqdm import tqdm

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('rag_system/output/logs/evaluate_embedding_models.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Define paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# TEST_QUERIES_PATH = BASE_DIR / "data" / "test_queries.json"
# OUTPUT_DIR = BASE_DIR / "output" / "evaluation"
# EMBEDDINGS_DIR = BASE_DIR / "data" / "processed" / "embeddings"
# INDEX_DIR = BASE_DIR / "retrieval" / "index"

# # Ensure output directory exists
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Embedding models to evaluate with their dimensions
# EMBEDDING_MODELS = {
#     "intfloat/e5-base-v2": 768,
#     "BAAI/bge-base-en-v1.5": 768,
#     "bge-large-en-v1.5": 1024,
#     "e5-large-v2": 1024
# }

# @contextmanager
# def track_memory():
#     """Track peak RAM usage during a block of code."""
#     process = psutil.Process()
#     peak_memory = 0
#     try:
#         yield
#         peak_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
#     finally:
#         logger.info(f"Peak RAM usage: {peak_memory:.2f} MB")

# def load_config():
#     """Load rag_config.yaml."""
#     try:
#         with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
#             return yaml.safe_load(f)
#     except Exception as e:
#         logger.error(f"Failed to load config: {e}")
#         raise

# def save_config(config):
#     """Save config to rag_config.yaml."""
#     try:
#         with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
#             yaml.safe_dump(config, f)
#     except Exception as e:
#         logger.error(f"Failed to save config: {e}")
#         raise

# def load_test_queries():
#     """Load test queries from JSON file."""
#     try:
#         with open(TEST_QUERIES_PATH, 'r', encoding='utf-8') as f:
#             queries = json.load(f)
#         logger.info(f"Loaded {len(queries)} test queries")
#         return queries
#     except Exception as e:
#         logger.error(f"Failed to load test queries: {e}")
#         raise

# def run_script(script_name, args=None):
#     """Run a Python script and measure execution time."""
#     # Handle different script locations
#     if script_name == "embedder.py":
#         script_path = BASE_DIR / "retrieval" / script_name
#     else:
#         script_path = BASE_DIR / "scripts" / script_name
    
#     # Ensure the script exists
#     if not script_path.exists():
#         logger.error(f"Script not found: {script_path}")
#         raise FileNotFoundError(f"Script not found: {script_path}")
    
#     cmd = ["python", str(script_path)]
#     if args:
#         cmd.extend(args)
    
#     logger.info(f"Running command: {' '.join(cmd)}")
#     logger.info(f"Working directory: {os.getcwd()}")
    
#     start_time = time.time()
#     try:
#         with track_memory():
#             # Add environment variables and proper working directory
#             env = os.environ.copy()
#             env['PYTHONPATH'] = str(BASE_DIR) + ':' + env.get('PYTHONPATH', '')
            
#             result = subprocess.run(
#                 cmd, 
#                 check=True, 
#                 capture_output=True, 
#                 text=True,
#                 cwd=str(BASE_DIR),  # Set working directory to BASE_DIR
#                 env=env
#             )
#         duration = time.time() - start_time
#         logger.info(f"Ran {script_name} in {duration:.2f} seconds")
#         logger.debug(f"Output: {result.stdout}")
#         if result.stderr:
#             logger.warning(f"Stderr: {result.stderr}")
#         return duration
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Failed to run {script_name}")
#         logger.error(f"Return code: {e.returncode}")
#         logger.error(f"Stdout: {e.stdout}")
#         logger.error(f"Stderr: {e.stderr}")
#         logger.error(f"Command: {' '.join(cmd)}")
#         raise

# def run_retrieval(query, top_k):
#     """Run retrieval and measure time and similarity scores."""
#     from rag_system.retrieval.retriever import retrieve
#     import asyncio
#     start_time = time.time()
#     with track_memory():
#         results = asyncio.run(retrieve(query, top_k))
#     duration = time.time() - start_time
#     similarities = [res['similarity'] for res in results]
#     mean_similarity = float(np.mean(similarities)) if similarities else 0.0
#     logger.info(f"Retrieval for query '{query[:30]}...': {len(results)} chunks, mean similarity: {mean_similarity:.4f}")
#     return results, duration, mean_similarity

# def run_generation(query, top_k):
#     """Run RAG pipeline for a query and measure generation time."""
#     script_path = BASE_DIR / "scripts" / "run_rag.py"
#     cmd = ["python", str(script_path), "--query", query, "--top_k", str(top_k)]
#     start_time = time.time()
#     try:
#         with track_memory():
#             result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#         duration = time.time() - start_time
#         output_file = OUTPUT_DIR / f"result_{''.join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')}.json"
#         with open(output_file, 'r', encoding='utf-8') as f:
#             result_data = json.load(f)
#         response = result_data['response']
#         logger.info(f"Generation for query '{query[:30]}...': {duration:.2f} seconds")
#         return response, duration
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Failed to run run_rag.py: {e.stderr}")
#         raise

# def main():
#     """Run MLflow experiment to evaluate embedding models."""
#     # Set up MLflow
#     mlflow.set_experiment("EmbeddingModelComparison")
    
#     # Load original config and queries
#     original_config = load_config()
#     test_queries = load_test_queries()
#     top_k = original_config['retrieval']['top_k']
    
#     # Back up original config
#     config_backup = CONFIG_PATH.with_suffix('.yaml.bak')
#     shutil.copy(CONFIG_PATH, config_backup)
#     logger.info(f"Backed up original config to {config_backup}")
    
#     # Change to the base directory to ensure relative paths work
#     original_cwd = os.getcwd()
#     os.chdir(BASE_DIR)
#     logger.info(f"Changed working directory to: {BASE_DIR}")
    
#     try:
#         for model_name, emb_dim in EMBEDDING_MODELS.items():
#             logger.info(f"Evaluating embedding model: {model_name}")
#             with mlflow.start_run(run_name=model_name):
#                 # Update config with model-specific settings
#                 config = load_config()
#                 config['retrieval']['embedding_model'] = model_name
#                 config['retrieval']['embedding_dimension'] = emb_dim
#                 save_config(config)
                
#                 try:
#                     # Log parameters
#                     mlflow.log_param("embedding_model", model_name)
#                     mlflow.log_param("embedding_dimension", emb_dim)
#                     mlflow.log_param("top_k", top_k)
#                     mlflow.log_param("batch_size", config['pipeline']['batch_size'])
                    
#                     # Clear previous embeddings and index
#                     if EMBEDDINGS_DIR.exists():
#                         shutil.rmtree(EMBEDDINGS_DIR)
#                         EMBEDDINGS_DIR.mkdir(parents=True)
#                     if INDEX_DIR.exists():
#                         shutil.rmtree(INDEX_DIR)
#                         INDEX_DIR.mkdir(parents=True)
                    
#                     # Run embedding generation
#                     logger.info("Running embedder.py...")
#                     embedding_time = run_script("embedder.py")
#                     mlflow.log_metric("embedding_time", embedding_time)
                    
#                     # Run index building
#                     logger.info("Running build_index.py...")
#                     index_time = run_script("build_index.py")
#                     mlflow.log_metric("index_build_time", index_time)
                    
#                     # Evaluate queries
#                     total_retrieval_time = 0
#                     total_generation_time = 0
#                     mean_similarities = []
#                     sample_responses = {}
                    
#                     for query in tqdm(test_queries, desc=f"Evaluating queries for {model_name}"):
#                         # Run retrieval
#                         results, retrieval_time, mean_similarity = run_retrieval(query, top_k)
#                         total_retrieval_time += retrieval_time
#                         mean_similarities.append(mean_similarity)
                        
#                         # Run generation
#                         response, generation_time = run_generation(query, top_k)
#                         total_generation_time += generation_time
#                         sample_responses[query] = response
                        
#                         # Log query-specific metrics
#                         safe_query_name = ''.join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
#                         mlflow.log_metric(f"retrieval_time_{safe_query_name}", retrieval_time)
#                         mlflow.log_metric(f"generation_time_{safe_query_name}", generation_time)
#                         mlflow.log_metric(f"similarity_{safe_query_name}", mean_similarity)
                    
#                     # Log aggregate metrics
#                     avg_retrieval_time = total_retrieval_time / len(test_queries)
#                     avg_generation_time = total_generation_time / len(test_queries)
#                     avg_similarity = float(np.mean(mean_similarities))
#                     mlflow.log_metric("avg_retrieval_time", avg_retrieval_time)
#                     mlflow.log_metric("avg_generation_time", avg_generation_time)
#                     mlflow.log_metric("avg_similarity_score", avg_similarity)
                    
#                     # Log sample responses as artifact
#                     responses_file = OUTPUT_DIR / f"responses_{model_name.replace('/', '_')}.json"
#                     with open(responses_file, 'w', encoding='utf-8') as f:
#                         json.dump(sample_responses, f, indent=2)
#                     mlflow.log_artifact(str(responses_file))
                    
#                     # Log config and logs
#                     mlflow.log_artifact(str(CONFIG_PATH))
#                     log_file = BASE_DIR / "rag_system" / "output" / "logs" / "evaluate_embedding_models.log"
#                     if log_file.exists():
#                         mlflow.log_artifact(str(log_file))
                    
#                     logger.info(f"Completed evaluation for {model_name}")
#                     logger.info(f"Average similarity: {avg_similarity:.4f}")
#                     logger.info(f"Average retrieval time: {avg_retrieval_time:.2f} seconds")
#                     logger.info(f"Average generation time: {avg_generation_time:.2f} seconds")
                
#                 except Exception as e:
#                     logger.error(f"Error evaluating {model_name}: {e}")
#                     # Log the error but continue with next model
#                     mlflow.log_param("evaluation_status", "failed")
#                     mlflow.log_param("error_message", str(e))
#                     continue
        
#     finally:
#         # Always restore original working directory and config
#         os.chdir(original_cwd)
#         logger.info(f"Restored working directory to: {original_cwd}")
        
#         # Restore original config after all evaluations
#         if config_backup.exists():
#             shutil.move(config_backup, CONFIG_PATH)
#             logger.info("Restored original config")
    
#     logger.info("Evaluation completed for all models")










# import os
# import json
# import yaml
# import logging
# import shutil
# import subprocess
# import time
# import numpy as np
# from pathlib import Path
# import mlflow
# import psutil
# from contextlib import contextmanager
# from tqdm import tqdm

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('rag_system/output/logs/evaluate_embedding_models.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Define paths - BASE_DIR should be the project root
# BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Go up to project root
# CONFIG_PATH = BASE_DIR / "rag_system" / "config" / "rag_config.yaml"
# TEST_QUERIES_PATH = BASE_DIR / "rag_system" / "data" / "test_queries.json"
# OUTPUT_DIR = BASE_DIR / "rag_system" / "output" / "evaluation"
# EMBEDDINGS_DIR = BASE_DIR / "rag_system" / "data" / "processed" / "embeddings"
# INDEX_DIR = BASE_DIR / "rag_system" / "retrieval" / "index"

# # Ensure output directory exists
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Embedding models to evaluate with their dimensions
# EMBEDDING_MODELS = {
#     "intfloat/e5-base-v2": 768,
#     "BAAI/bge-base-en-v1.5": 768,
#     "bge-large-en-v1.5": 1024,
#     "e5-large-v2": 1024
# }

# @contextmanager
# def track_memory():
#     """Track peak RAM usage during a block of code."""
#     process = psutil.Process()
#     peak_memory = 0
#     try:
#         yield
#         peak_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
#     finally:
#         logger.info(f"Peak RAM usage: {peak_memory:.2f} MB")

# def load_config():
#     """Load rag_config.yaml."""
#     try:
#         with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
#             return yaml.safe_load(f)
#     except Exception as e:
#         logger.error(f"Failed to load config: {e}")
#         raise

# def save_config(config):
#     """Save config to rag_config.yaml."""
#     try:
#         with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
#             yaml.safe_dump(config, f)
#     except Exception as e:
#         logger.error(f"Failed to save config: {e}")
#         raise

# def load_test_queries():
#     """Load test queries from JSON file."""
#     try:
#         with open(TEST_QUERIES_PATH, 'r', encoding='utf-8') as f:
#             queries = json.load(f)
#         logger.info(f"Loaded {len(queries)} test queries")
#         return queries
#     except Exception as e:
#         logger.error(f"Failed to load test queries: {e}")
#         raise

# def run_script(script_name, args=None):
#     """Run a Python script and measure execution time."""
#     # Handle different script locations
#     if script_name == "embedder.py":
#         script_path = BASE_DIR / "rag_system" / "retrieval" / script_name
#     else:
#         script_path = BASE_DIR / "rag_system" / "scripts" / script_name
    
#     # Ensure the script exists
#     if not script_path.exists():
#         logger.error(f"Script not found: {script_path}")
#         raise FileNotFoundError(f"Script not found: {script_path}")
    
#     cmd = ["python", str(script_path)]
#     if args:
#         cmd.extend(args)
    
#     logger.info(f"Running command: {' '.join(cmd)}")
#     logger.info(f"Working directory: {BASE_DIR}")
    
#     start_time = time.time()
#     try:
#         with track_memory():
#             env = os.environ.copy()
#             env['PYTHONPATH'] = str(BASE_DIR) + ':' + env.get('PYTHONPATH', '')
#             result = subprocess.run(
#                 cmd, 
#                 check=True, 
#                 capture_output=True, 
#                 text=True,
#                 cwd=str(BASE_DIR),
#                 env=env
#             )
#         duration = time.time() - start_time
#         logger.info(f"Ran {script_name} in {duration:.2f} seconds")
#         logger.debug(f"Output: {result.stdout}")
#         if result.stderr:
#             logger.warning(f"Stderr: {result.stderr}")
#         return duration
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Failed to run {script_name}")
#         logger.error(f"Return code: {e.returncode}")
#         logger.error(f"Stdout: {e.stdout}")
#         logger.error(f"Stderr: {e.stderr}")
#         logger.error(f"Command: {' '.join(cmd)}")
#         raise

# def run_retrieval(query, top_k):
#     """Run retrieval and measure time and similarity scores."""
#     from rag_system.retrieval.retriever import retrieve
#     import asyncio
#     start_time = time.time()
#     with track_memory():
#         results = asyncio.run(retrieve(query, top_k))
#     duration = time.time() - start_time
#     similarities = [res['similarity'] for res in results]
#     mean_similarity = float(np.mean(similarities)) if similarities else 0.0
#     logger.info(f"Retrieval for query '{query[:30]}...': {len(results)} chunks, mean similarity: {mean_similarity:.4f}")
#     return results, duration, mean_similarity

# def run_generation(query, top_k):
#     """Run RAG pipeline for a query and measure generation time."""
#     script_path = BASE_DIR / "rag_system" / "scripts" / "run_rag.py"
#     cmd = ["python", str(script_path), "--query", query, "--top_k", str(top_k)]
#     start_time = time.time()
#     try:
#         with track_memory():
#             env = os.environ.copy()
#             env['PYTHONPATH'] = str(BASE_DIR) + ':' + env.get('PYTHONPATH', '')
#             result = subprocess.run(
#                 cmd, 
#                 check=True, 
#                 capture_output=True, 
#                 text=True,
#                 cwd=str(BASE_DIR),
#                 env=env
#             )
#         duration = time.time() - start_time
#         output_file = OUTPUT_DIR / f"result_{''.join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')}.json"
#         with open(output_file, 'r', encoding='utf-8') as f:
#             result_data = json.load(f)
#         response = result_data['response']
#         logger.info(f"Generation for query '{query[:30]}...': {duration:.2f} seconds")
#         return response, duration
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Failed to run run_rag.py: {e.stderr}")
#         raise

# def check_embeddings_exist():
#     """Check if all required embedding files exist."""
#     required_files = ['anime.npy', 'bollywood.npy', 'hollywood.npy', 'kdrama.npy', 'kmovie.npy', 'manga.npy']
#     return all((EMBEDDINGS_DIR / f).exists() for f in required_files)

# def main():
#     """Run MLflow experiment to evaluate embedding models."""
#     # Set MLflow tracking URI to root/mlruns
#     mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
#     mlflow.set_experiment("EmbeddingModelComparison")
    
#     # Load original config and queries
#     original_config = load_config()
#     test_queries = load_test_queries()
#     top_k = original_config['retrieval']['top_k']
    
#     # Back up original config
#     config_backup = CONFIG_PATH.with_suffix('.yaml.bak')
#     shutil.copy(CONFIG_PATH, config_backup)
#     logger.info(f"Backed up original config to {config_backup}")
#     logger.info(f"Working directory: {BASE_DIR}")
    
#     try:
#         for model_name, emb_dim in EMBEDDING_MODELS.items():
#             logger.info(f"Evaluating embedding model: {model_name}")
#             with mlflow.start_run(run_name=model_name):
#                 # Update config with model-specific settings
#                 config = load_config()
#                 config['retrieval']['embedding_model'] = model_name
#                 config['retrieval']['embedding_dimension'] = emb_dim
#                 save_config(config)
                
#                 try:
#                     # Log parameters
#                     mlflow.log_param("embedding_model", model_name)
#                     mlflow.log_param("embedding_dimension", emb_dim)
#                     mlflow.log_param("top_k", top_k)
#                     mlflow.log_param("batch_size", config['pipeline']['batch_size'])
                    
#                     # Handle embeddings
#                     if model_name == "intfloat/e5-base-v2" and check_embeddings_exist():
#                         logger.info("Reusing existing embeddings for intfloat/e5-base-v2")
#                         embedding_time = 0  # No time since reused
#                     else:
#                         logger.info("Generating new embeddings")
#                         if EMBEDDINGS_DIR.exists():
#                             shutil.rmtree(EMBEDDINGS_DIR)
#                         EMBEDDINGS_DIR.mkdir(parents=True)
#                         embedding_time = run_script("embedder.py")
#                         if not check_embeddings_exist():
#                             raise RuntimeError("Embedding generation failed: missing required .npy files")
#                     mlflow.log_metric("embedding_time", embedding_time)
                    
#                     # Clear index for new model
#                     if INDEX_DIR.exists():
#                         shutil.rmtree(INDEX_DIR)
#                     INDEX_DIR.mkdir(parents=True)
                    
#                     # Run index building
#                     logger.info("Running build_index.py...")
#                     index_time = run_script("build_index.py")
#                     mlflow.log_metric("index_build_time", index_time)
                    
#                     # Evaluate queries
#                     total_retrieval_time = 0
#                     total_generation_time = 0
#                     mean_similarities = []
#                     sample_responses = {}
                    
#                     for query in tqdm(test_queries, desc=f"Evaluating queries for {model_name}"):
#                         results, retrieval_time, mean_similarity = run_retrieval(query, top_k)
#                         total_retrieval_time += retrieval_time
#                         mean_similarities.append(mean_similarity)
                        
#                         response, generation_time = run_generation(query, top_k)
#                         total_generation_time += generation_time
#                         sample_responses[query] = response
                        
#                         safe_query_name = ''.join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
#                         mlflow.log_metric(f"retrieval_time_{safe_query_name}", retrieval_time)
#                         mlflow.log_metric(f"generation_time_{safe_query_name}", generation_time)
#                         mlflow.log_metric(f"similarity_{safe_query_name}", mean_similarity)
                    
#                     # Log aggregate metrics
#                     avg_retrieval_time = total_retrieval_time / len(test_queries)
#                     avg_generation_time = total_generation_time / len(test_queries)
#                     avg_similarity = float(np.mean(mean_similarities))
#                     mlflow.log_metric("avg_retrieval_time", avg_retrieval_time)
#                     mlflow.log_metric("avg_generation_time", avg_generation_time)
#                     mlflow.log_metric("avg_similarity_score", avg_similarity)
                    
#                     # Log sample responses as artifact
#                     responses_file = OUTPUT_DIR / f"responses_{model_name.replace('/', '_')}.json"
#                     with open(responses_file, 'w', encoding='utf-8') as f:
#                         json.dump(sample_responses, f, indent=2)
#                     mlflow.log_artifact(str(responses_file))
                    
#                     # Log config and logs
#                     mlflow.log_artifact(str(CONFIG_PATH))
#                     log_file = BASE_DIR / "rag_system" / "output" / "logs" / "evaluate_embedding_models.log"
#                     if log_file.exists():
#                         mlflow.log_artifact(str(log_file))
                    
#                     logger.info(f"Completed evaluation for {model_name}")
#                     logger.info(f"Average similarity: {avg_similarity:.4f}")
#                     logger.info(f"Average retrieval time: {avg_retrieval_time:.2f} seconds")
#                     logger.info(f"Average generation time: {avg_generation_time:.2f} seconds")
                
#                 except Exception as e:
#                     logger.error(f"Error evaluating {model_name}: {e}")
#                     mlflow.log_param("evaluation_status", "failed")
#                     mlflow.log_param("error_message", str(e))
#                     continue
        
#     finally:
#         if config_backup.exists():
#             shutil.move(config_backup, CONFIG_PATH)
#             logger.info("Restored original config")
    
#     logger.info("Evaluation completed for all models")

# if __name__ == "__main__":
#     main()











import os
import sys
import json
import yaml
import logging
import shutil
import subprocess
import time
import numpy as np
from pathlib import Path
import mlflow
import psutil
from contextlib import contextmanager
from tqdm import tqdm

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Go up to project root
sys.path.insert(0, str(BASE_DIR))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system/output/logs/evaluate_embedding_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
CONFIG_PATH = BASE_DIR / "rag_system" / "config" / "rag_config.yaml"
TEST_QUERIES_PATH = BASE_DIR / "rag_system" / "data" / "test_queries.json"
OUTPUT_DIR = BASE_DIR / "rag_system" / "output" / "evaluation"
EMBEDDINGS_DIR = BASE_DIR / "rag_system" / "data" / "processed" / "embeddings"
INDEX_DIR = BASE_DIR / "rag_system" / "retrieval" / "index"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding models to evaluate with their dimensions
EMBEDDING_MODELS = {
    "intfloat/e5-base-v2": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-large-v2": 1024
}

@contextmanager
def track_memory():
    """Track peak RAM usage during a block of code."""
    process = psutil.Process()
    peak_memory = 0
    try:
        yield
        peak_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    finally:
        logger.info(f"Peak RAM usage: {peak_memory:.2f} MB")

def load_config():
    """Load rag_config.yaml."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def save_config(config):
    """Save config to rag_config.yaml."""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise

def load_test_queries():
    """Load test queries from JSON file."""
    try:
        with open(TEST_QUERIES_PATH, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} test queries")
        return queries
    except Exception as e:
        logger.error(f"Failed to load test queries: {e}")
        raise

def run_script(script_name, args=None):
    """Run a Python script and measure execution time."""
    # Handle different script locations
    if script_name == "embedder.py":
        script_path = BASE_DIR / "rag_system" / "retrieval" / script_name
    else:
        script_path = BASE_DIR / "rag_system" / "scripts" / script_name
    
    # Ensure the script exists
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    cmd = ["python", str(script_path)]
    if args:
        cmd.extend(args)
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Working directory: {BASE_DIR}")
    
    start_time = time.time()
    try:
        with track_memory():
            env = os.environ.copy()
            # Set PYTHONPATH to include project root
            env['PYTHONPATH'] = str(BASE_DIR) + ':' + env.get('PYTHONPATH', '')
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=str(BASE_DIR),
                env=env
            )
        duration = time.time() - start_time
        logger.info(f"Ran {script_name} in {duration:.2f} seconds")
        logger.debug(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run {script_name}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Command: {' '.join(cmd)}")
        raise

def run_retrieval(query, top_k):
    """Run retrieval and measure time and similarity scores."""
    try:
        # Import here after path is set
        from rag_system.retrieval.retriever import retrieve
        import asyncio
        
        start_time = time.time()
        with track_memory():
            results = asyncio.run(retrieve(query, top_k))
        duration = time.time() - start_time
        similarities = [res['similarity'] for res in results]
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        logger.info(f"Retrieval for query '{query[:30]}...': {len(results)} chunks, mean similarity: {mean_similarity:.4f}")
        return results, duration, mean_similarity
    except Exception as e:
        logger.error(f"Error in run_retrieval: {e}")
        raise

def run_generation(query, top_k):
    """Run RAG pipeline for a query and measure generation time."""
    script_path = BASE_DIR / "rag_system" / "scripts" / "run_rag.py"
    cmd = ["python", str(script_path), "--query", query, "--top_k", str(top_k)]
    start_time = time.time()
    try:
        with track_memory():
            env = os.environ.copy()
            # Set PYTHONPATH to include project root
            env['PYTHONPATH'] = str(BASE_DIR) + ':' + env.get('PYTHONPATH', '')
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=str(BASE_DIR),
                env=env
            )
        duration = time.time() - start_time
        
        # Load config to get the correct results directory
        config = load_config()
        results_dir = config["pipeline"]["output"]["results_dir"]
        
        # Create safe query name
        safe_query = "".join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        
        # Check the correct location based on config
        output_file = BASE_DIR / "rag_system" / results_dir / f"result_{safe_query}.json"
        
        result_data = None
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        else:
            logger.warning(f"Could not find output file at: {output_file}")
            return "No response generated", duration
            
        response = result_data.get('response', 'No response found')
        logger.info(f"Generation for query '{query[:30]}...': {duration:.2f} seconds")
        return response, duration
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run run_rag.py: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error in run_generation: {e}")
        raise

def check_embeddings_exist():
    """Check if all required embedding files exist."""
    required_files = ['anime.npy', 'bollywood.npy', 'hollywood.npy', 'kdrama.npy', 'kmovie.npy', 'manga.npy']
    return all((EMBEDDINGS_DIR / f).exists() for f in required_files)

def main():
    """Run MLflow experiment to evaluate embedding models."""
    # Set MLflow tracking URI to root/mlruns
    mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
    mlflow.set_experiment("EmbeddingModelComparison")
    
    # Load original config and queries
    original_config = load_config()
    test_queries = load_test_queries()
    top_k = original_config['retrieval']['top_k']
    
    # Back up original config
    config_backup = CONFIG_PATH.with_suffix('.yaml.bak')
    shutil.copy(CONFIG_PATH, config_backup)
    logger.info(f"Backed up original config to {config_backup}")
    logger.info(f"Working directory: {BASE_DIR}")
    
    try:
        for model_name, emb_dim in EMBEDDING_MODELS.items():
            logger.info(f"Evaluating embedding model: {model_name}")
            with mlflow.start_run(run_name=model_name):
                # Update config with model-specific settings
                config = load_config()
                config['retrieval']['embedding_model'] = model_name
                config['retrieval']['embedding_dimension'] = emb_dim
                save_config(config)
                
                try:
                    # Log parameters
                    mlflow.log_param("embedding_model", model_name)
                    mlflow.log_param("embedding_dimension", emb_dim)
                    mlflow.log_param("top_k", top_k)
                    mlflow.log_param("batch_size", config['pipeline']['batch_size'])
                    
                    # Handle embeddings
                    if model_name == "intfloat/e5-base-v2" and check_embeddings_exist():
                        logger.info("Reusing existing embeddings for intfloat/e5-base-v2")
                        embedding_time = 0  # No time since reused
                    else:
                        logger.info("Generating new embeddings")
                        if EMBEDDINGS_DIR.exists():
                            shutil.rmtree(EMBEDDINGS_DIR)
                        EMBEDDINGS_DIR.mkdir(parents=True)
                        embedding_time = run_script("embedder.py")
                        if not check_embeddings_exist():
                            raise RuntimeError("Embedding generation failed: missing required .npy files")
                    mlflow.log_metric("embedding_time", embedding_time)
                    
                    # Clear index for new model
                    if INDEX_DIR.exists():
                        shutil.rmtree(INDEX_DIR)
                    INDEX_DIR.mkdir(parents=True)
                    
                    # Run index building
                    logger.info("Running build_index.py...")
                    index_time = run_script("build_index.py")
                    mlflow.log_metric("index_build_time", index_time)
                    
                    # Evaluate queries
                    total_retrieval_time = 0
                    total_generation_time = 0
                    mean_similarities = []
                    sample_responses = {}
                    
                    for query in tqdm(test_queries, desc=f"Evaluating queries for {model_name}"):
                        try:
                            results, retrieval_time, mean_similarity = run_retrieval(query, top_k)
                            total_retrieval_time += retrieval_time
                            mean_similarities.append(mean_similarity)
                            
                            response, generation_time = run_generation(query, top_k)
                            total_generation_time += generation_time
                            sample_responses[query] = response
                            
                            safe_query_name = ''.join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
                            mlflow.log_metric(f"retrieval_time_{safe_query_name}", retrieval_time)
                            mlflow.log_metric(f"generation_time_{safe_query_name}", generation_time)
                            mlflow.log_metric(f"similarity_{safe_query_name}", mean_similarity)
                        except Exception as e:
                            logger.error(f"Error processing query '{query}': {e}")
                            continue
                    
                    # Log aggregate metrics
                    if len(test_queries) > 0:
                        avg_retrieval_time = total_retrieval_time / len(test_queries)
                        avg_generation_time = total_generation_time / len(test_queries)
                        avg_similarity = float(np.mean(mean_similarities)) if mean_similarities else 0.0
                        mlflow.log_metric("avg_retrieval_time", avg_retrieval_time)
                        mlflow.log_metric("avg_generation_time", avg_generation_time)
                        mlflow.log_metric("avg_similarity_score", avg_similarity)
                        
                        # Log sample responses as artifact
                        responses_file = OUTPUT_DIR / f"responses_{model_name.replace('/', '_')}.json"
                        with open(responses_file, 'w', encoding='utf-8') as f:
                            json.dump(sample_responses, f, indent=2, ensure_ascii=False)
                        mlflow.log_artifact(str(responses_file))
                        
                        logger.info(f"Completed evaluation for {model_name}")
                        logger.info(f"Average similarity: {avg_similarity:.4f}")
                        logger.info(f"Average retrieval time: {avg_retrieval_time:.2f} seconds")
                        logger.info(f"Average generation time: {avg_generation_time:.2f} seconds")
                    
                    # Log config and logs
                    mlflow.log_artifact(str(CONFIG_PATH))
                    log_file = BASE_DIR / "rag_system" / "output" / "logs" / "evaluate_embedding_models.log"
                    if log_file.exists():
                        mlflow.log_artifact(str(log_file))
                    
                    mlflow.log_param("evaluation_status", "success")
                
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    mlflow.log_param("evaluation_status", "failed")
                    mlflow.log_param("error_message", str(e))
                    continue
        
    finally:
        if config_backup.exists():
            shutil.move(config_backup, CONFIG_PATH)
            logger.info("Restored original config")
    
    logger.info("Evaluation completed for all models")

if __name__ == "__main__":
    main()