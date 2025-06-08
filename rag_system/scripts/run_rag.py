# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import sys
# from pathlib import Path
# import yaml
# import logging
# import json
# import argparse
# from retrieval.retriever import retrieve
# from generation.generator import Generator

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# # Load config
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Logging setup
# log_path = BASE_DIR / config["pipeline"]["logging"]["log_dir"] / config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def run_rag(query, top_k, max_new_tokens):
#     logger.info(f"Processing query: {query}")
#     try:
#         # Retrieve chunks
#         chunks = retrieve(query, top_k)
#         logger.info(f"Retrieved {len(chunks)} chunks")
        
#         # Generate response
#         generator = Generator()
#         response = generator.generate(query, chunks, max_new_tokens=max_new_tokens)
        
#         # Save result
#         result = {"query": query, "response": response, "retrieved_chunks": chunks}
#         output_dir = BASE_DIR / config["pipeline"]["output"]["results_dir"]
#         output_dir.mkdir(parents=True, exist_ok=True)
#         output_file = output_dir / f"result_{query[:20].replace(' ', '_')}.json"
#         with open(output_file, "w") as f:
#             json.dump(result, f, indent=2)
#         logger.info(f"Saved result to {output_file}")
#         return response
#     except Exception as e:
#         logger.error(f"RAG pipeline failed: {e}")
#         return f"Error processing query: {e}"

# def main():
#     parser = argparse.ArgumentParser(description="Interactive RAG pipeline for entertainment queries")
#     parser.add_argument("--query", help="Query for the RAG system")
#     parser.add_argument("--top_k", type=int, default=config["retrieval"]["top_k"], help="Number of chunks to retrieve")
#     parser.add_argument("--max_new_tokens", type=int, default=config["generation"]["parameters"]["max_new_tokens"], help="Maximum new tokens for generation")
#     args = parser.parse_args()

#     # Interactive mode if no query provided
#     if not args.query:
#         print("Enter your query (or 'quit' to exit):")
#         while True:
#             query = input("> ").strip()
#             if query.lower() == "quit":
#                 break
#             if query:
#                 response = run_rag(query, args.top_k, args.max_new_tokens)
#                 print(f"\nResponse:\n{response}\n")
#             print("Enter another query (or 'quit' to exit):")
#     else:
#         response = run_rag(args.query, args.top_k, args.max_new_tokens)
#         print(f"\nResponse:\n{response}\n")

# if __name__ == "__main__":
#     main()











# import sys
# from pathlib import Path

# # Setup paths first
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# import yaml
# import logging
# import json
# import argparse
# from retrieval.retriever import retrieve
# from generation.generator import Generator

# # Load config
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Logging setup
# log_path = BASE_DIR / config["pipeline"]["logging"]["log_dir"] / config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize generator once globally to avoid reloading model for each query
# generator = None

# def get_generator():
#     """Get or create the global generator instance"""
#     global generator
#     if generator is None:
#         logger.info("Initializing Generator (first time)...")
#         generator = Generator()
#     return generator

# def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
#     """Run the RAG pipeline with proper parameter handling"""
#     logger.info(f"Processing query: {query}")
    
#     # Use config defaults if parameters not provided
#     top_k = top_k or config["retrieval"]["top_k"]
#     max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
#     temperature = temperature or config["generation"]["parameters"]["temperature"]
#     top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
#     try:
#         # Retrieve chunks
#         logger.info(f"Retrieving top {top_k} chunks...")
#         chunks = retrieve(query, top_k)
#         logger.info(f"Retrieved {len(chunks)} chunks")
        
#         # Generate response using the global generator
#         gen = get_generator()
#         logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
#         response = gen.generate(
#             query=query,
#             retrieved_chunks=chunks,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k_sampling
#         )
        
#         # Save result
#         result = {
#             "query": query,
#             "response": response,
#             "retrieved_chunks": chunks,
#             "parameters": {
#                 "top_k_retrieval": top_k,
#                 "max_new_tokens": max_new_tokens,
#                 "temperature": temperature,
#                 "top_k_sampling": top_k_sampling
#             }
#         }
        
#         output_dir = BASE_DIR / config["pipeline"]["output"]["results_dir"]
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Create a safe filename
#         safe_query = "".join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
#         output_file = output_dir / f"result_{safe_query}.json"
        
#         with open(output_file, "w") as f:
#             json.dump(result, f, indent=2)
#         logger.info(f"Saved result to {output_file}")
        
#         return response
        
#     except Exception as e:
#         logger.exception(f"RAG pipeline failed: {e}")  # Use exception() to get full traceback
#         return f"Error processing query: {e}"

# def main():
#     parser = argparse.ArgumentParser(description="Interactive RAG pipeline for entertainment queries")
#     parser.add_argument("--query", help="Query for the RAG system")
#     parser.add_argument("--top_k", type=int, default=None, help=f"Number of chunks to retrieve (default: {config['retrieval']['top_k']})")
#     parser.add_argument("--max_new_tokens", type=int, default=None, help=f"Maximum new tokens for generation (default: {config['generation']['parameters']['max_new_tokens']})")
#     parser.add_argument("--temperature", type=float, default=None, help=f"Temperature for generation (default: {config['generation']['parameters']['temperature']})")
#     parser.add_argument("--top_k_sampling", type=int, default=None, help=f"Top-k for sampling (default: {config['generation']['parameters'].get('top_k', 50)})")
    
#     args = parser.parse_args()
    
#     # Initialize generator once at startup
#     logger.info("Starting RAG pipeline...")
#     try:
#         get_generator()  # This will initialize it
#         logger.info("Generator initialized successfully!")
#     except Exception as e:
#         logger.error(f"Failed to initialize generator: {e}")
#         return
    
#     # Interactive mode if no query provided
#     if not args.query:
#         print("🎬 Entertainment RAG System Ready!")
#         print("Ask me about movies, anime, manga, K-dramas, or Bollywood!")
#         print("Type 'quit' to exit.\n")
        
#         while True:
#             try:
#                 query = input("🎭 Your query > ").strip()
#                 if query.lower() in ["quit", "exit", "q"]:
#                     print("Goodbye! 👋")
#                     break
                    
#                 if query:
#                     print(f"\n🔍 Processing: {query}")
#                     response = run_rag(
#                         query=query,
#                         top_k=args.top_k,
#                         max_new_tokens=args.max_new_tokens,
#                         temperature=args.temperature,
#                         top_k_sampling=args.top_k_sampling
#                     )
#                     print(f"\n🎯 Response:\n{response}\n")
#                     print("-" * 60)
#                 else:
#                     print("Please enter a valid query.")
                    
#             except KeyboardInterrupt:
#                 print("\n\nGoodbye! 👋")
#                 break
#             except Exception as e:
#                 logger.error(f"Error in interactive mode: {e}")
#                 print(f"Error: {e}")
#     else:
#         # Single query mode
#         print(f"🔍 Processing query: {args.query}")
#         response = run_rag(
#             query=args.query,
#             top_k=args.top_k,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_k_sampling=args.top_k_sampling
#         )
#         print(f"\n🎯 Response:\n{response}")

# if __name__ == "__main__":
#     main()
















# original-------------------------------------------------------------------------------------------------------




# import sys
# from pathlib import Path

# # Setup paths first
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# import yaml
# import logging
# import json
# import argparse
# from retrieval.retriever import retrieve
# from generation.generator import Generator
# from memory.conversation_memory import ConversationMemory

# # Load config
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Logging setup
# log_path = BASE_DIR / config["pipeline"]["logging"]["log_dir"] / config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize generator and memory once globally to avoid reloading model for each query
# generator = None
# memory = ConversationMemory(max_turns=config["generation"].get("history_window", 10))

# def get_generator():
#     """Get or create the global generator instance"""
#     global generator
#     if generator is None:
#         logger.info("Initializing Generator (first time)...")
#         generator = Generator()
#     return generator

# def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
#     """Run the RAG pipeline with proper parameter handling"""
#     logger.info(f"Processing query: {query}")
    
#     # Use config defaults if parameters not provided
#     top_k = top_k or config["retrieval"]["top_k"]
#     max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
#     temperature = temperature or config["generation"]["parameters"]["temperature"]
#     top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
#     try:
#         # Retrieve chunks
#         logger.info(f"Retrieving top {top_k} chunks...")
#         chunks = retrieve(query, top_k)
#         logger.info(f"Retrieved {len(chunks)} chunks")
        
#         # Generate response using the global generator with memory
#         gen = get_generator()
#         logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
#         response = gen.generate(
#             query=query,
#             retrieved_chunks=chunks,
#             memory=memory,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k_sampling
#         )
        
#         # Store the conversation turn in memory
#         memory.add_turn(query, response)
        
#         # Save result
#         result = {
#             "query": query,
#             "response": response,
#             "retrieved_chunks": chunks,
#             "parameters": {
#                 "top_k_retrieval": top_k,
#                 "max_new_tokens": max_new_tokens,
#                 "temperature": temperature,
#                 "top_k_sampling": top_k_sampling
#             }
#         }
        
#         output_dir = BASE_DIR / config["pipeline"]["output"]["results_dir"]
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Create a safe filename
#         safe_query = "".join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
#         output_file = output_dir / f"result_{safe_query}.json"
        
#         with open(output_file, "w") as f:
#             json.dump(result, f, indent=2)
#         logger.info(f"Saved result to {output_file}")
        
#         return response
        
#     except Exception as e:
#         logger.exception(f"RAG pipeline failed: {e}")  # Use exception() to get full traceback
#         return f"Error processing query: {e}"

# def main():
#     parser = argparse.ArgumentParser(description="Interactive RAG pipeline for entertainment queries")
#     parser.add_argument("--query", help="Query for the RAG system")
#     parser.add_argument("--top_k", type=int, default=None, help=f"Number of chunks to retrieve (default: {config['retrieval']['top_k']})")
#     parser.add_argument("--max_new_tokens", type=int, default=None, help=f"Maximum new tokens for generation (default: {config['generation']['parameters']['max_new_tokens']})")
#     parser.add_argument("--temperature", type=float, default=None, help=f"Temperature for generation (default: {config['generation']['parameters']['temperature']})")
#     parser.add_argument("--top_k_sampling", type=int, default=None, help=f"Top-k for sampling (default: {config['generation']['parameters'].get('top_k', 50)})")
    
#     args = parser.parse_args()
    
#     # Initialize generator once at startup
#     logger.info("Starting RAG pipeline...")
#     try:
#         get_generator()  # This will initialize it
#         logger.info("Generator initialized successfully!")
#     except Exception as e:
#         logger.error(f"Failed to initialize generator: {e}")
#         return
    
#     # Interactive mode if no query provided
#     if not args.query:
#         print("🎬 Entertainment RAG System Ready!")
#         print("Ask me about movies, anime, manga, K-dramas, or Bollywood!")
#         print("Type 'quit' to exit.\n")
        
#         while True:
#             try:
#                 query = input("🎭 Your query > ").strip()
#                 if query.lower() in ["quit", "exit", "q"]:
#                     print("Goodbye! 👋")
#                     break
                    
#                 if query:
#                     print(f"\n🔍 Processing: {query}")
#                     response = run_rag(
#                         query=query,
#                         top_k=args.top_k,
#                         max_new_tokens=args.max_new_tokens,
#                         temperature=args.temperature,
#                         top_k_sampling=args.top_k_sampling
#                     )
#                     print(f"\n🎯 Response:\n{response}\n")
#                     print("-" * 60)
#                 else:
#                     print("Please enter a valid query.")
                    
#             except KeyboardInterrupt:
#                 print("\n\nGoodbye! 👋")
#                 break
#             except Exception as e:
#                 logger.error(f"Error in interactive mode: {e}")
#                 print(f"Error: {e}")
#     else:
#         # Single query mode
#         print(f"🔍 Processing query: {args.query}")
#         response = run_rag(
#             query=args.query,
#             top_k=args.top_k,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_k_sampling=args.top_k_sampling
#         )
#         print(f"\n🎯 Response:\n{response}")

# if __name__ == "__main__":
#     main()






# import sys
# from pathlib import Path
# import yaml
# import logging
# import json
# import argparse
# import asyncio
# from retrieval.retriever import retrieve
# from generation.generator import Generator
# from memory.conversation_memory import ConversationMemory
# from scripts.reddit_scraper import SmartRedditScraperBotWithEmbeddings

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# # Load config
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Logging setup
# log_path = BASE_DIR / config["pipeline"]["logging"]["log_dir"] / config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize generator and memory
# generator = None
# memory = ConversationMemory(max_turns=config["generation"].get("history_window", 10))
# reddit_triggers = ["reddit", "discussion", "people think", "community", "opinions", "users say"]

# def detect_query_type(query):
#     """Detect query type: factual, opinion, or recommendation"""
#     query_lower = query.lower()
#     factual_keywords = ["plot", "story", "ending", "summary", "explain", "what happens", "details"]
#     opinion_keywords = ["think", "feel", "opinion", "view", "believe", "discussion", "users say"]
#     recommendation_keywords = ["recommend", "suggest", "similar to", "like", "best", "top"]
    
#     if any(keyword in query_lower for keyword in recommendation_keywords):
#         return "recommendation"
#     elif any(keyword in query_lower for keyword in opinion_keywords):
#         return "opinion"
#     elif any(keyword in query_lower for keyword in factual_keywords):
#         return "factual"
#     return "factual"  # Default to factual

# async def retrieve_vector(query, top_k):
#     """Retrieve vector store chunks asynchronously"""
#     logger.info(f"Retrieving top {top_k} chunks from vector store...")
#     chunks = retrieve(query, top_k)  # Assumes retrieve is sync; adapt if async
#     logger.info(f"Retrieved {len(chunks)} vector store chunks")
#     return chunks

# async def retrieve_reddit(query):
#     """Retrieve Reddit data asynchronously"""
#     logger.info("Triggering Reddit scraper...")
#     scraper = SmartRedditScraperBotWithEmbeddings()
#     result = await asyncio.to_thread(scraper.search_and_preprocess_with_embeddings, query, num_posts=3)
#     reddit_data = result.get('data', [])
#     logger.info(f"Retrieved {len(reddit_data)} Reddit items")
#     return reddit_data

# def get_generator():
#     """Get or create the global generator instance"""
#     global generator
#     if generator is None:
#         logger.info("Initializing Generator (first time)...")
#         generator = Generator()
#     return generator

# async def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
#     """Run the RAG pipeline with Reddit integration"""
#     logger.info(f"Processing query: {query}")
    
#     # Use config defaults
#     top_k = top_k or config["retrieval"]["top_k"]
#     max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
#     temperature = temperature or config["generation"]["parameters"]["temperature"]
#     top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
#     try:
#         # Detect query type and Reddit trigger
#         query_type = detect_query_type(query)
#         use_reddit = any(trigger in query.lower() for trigger in reddit_triggers)
#         vector_chunks = []
#         reddit_data = []
        
#         # Parallel retrieval
#         if use_reddit:
#             tasks = [
#                 retrieve_vector(query, top_k),
#                 retrieve_reddit(query)
#             ]
#             vector_chunks, reddit_data = await asyncio.gather(*tasks, return_exceptions=True)
#         else:
#             vector_chunks = await retrieve_vector(query, top_k)
        
#         # Handle potential errors
#         if isinstance(vector_chunks, Exception):
#             logger.error(f"Vector retrieval failed: {vector_chunks}")
#             vector_chunks = []
#         if isinstance(reddit_data, Exception):
#             logger.error(f"Reddit retrieval failed: {reddit_data}")
#             reddit_data = []
        
#         # Check vector store similarity
#         use_only_reddit = False
#         if vector_chunks and not reddit_data:
#             max_similarity = max(1 - chunk['distance'] for chunk in vector_chunks)
#             if max_similarity < 0.5:
#                 logger.info("Vector store similarity < 0.5, but no Reddit data available")
        
#         if vector_chunks and reddit_data:
#             max_similarity = max(1 - chunk['distance'] for chunk in vector_chunks)
#             if max_similarity < 0.5:
#                 logger.info("Vector store similarity < 0.5, using Reddit data only")
#                 use_only_reddit = True
        
#         # Prepare context
#         vector_context = [chunk['text'] for chunk in vector_chunks] if not use_only_reddit else []
#         reddit_context = [item['text'] for item in reddit_data] if reddit_data else []
        
#         # Generate response
#         gen = get_generator()
#         logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
#         response = gen.generate(
#             query=query,
#             vector_context=vector_context,
#             reddit_context=reddit_context,
#             query_type=query_type,
#             memory=memory,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k_sampling
#         )
        
#         # Store conversation turn
#         memory.add_turn(query, response)
        
#         # Save result
#         result = {
#             "query": query,
#             "response": response,
#             "vector_chunks": vector_chunks,
#             "reddit_data": reddit_data,
#             "query_type": query_type,
#             "parameters": {
#                 "top_k_retrieval": top_k,
#                 "max_new_tokens": max_new_tokens,
#                 "temperature": temperature,
#                 "top_k_sampling": top_k_sampling,
#                 "used_reddit": use_reddit,
#                 "used_only_reddit": use_only_reddit
#             }
#         }
        
#         output_dir = BASE_DIR / config["pipeline"]["output"]["results_dir"]
#         output_dir.mkdir(parents=True, exist_ok=True)
#         safe_query = "".join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
#         output_file = output_dir / f"result_{safe_query}.json"
        
#         with open(output_file, "w") as f:
#             json.dump(result, f, indent=2)
#         logger.info(f"Saved result to {output_file}")
        
#         return response
        
#     except Exception as e:
#         logger.exception(f"RAG pipeline failed: {e}")
#         return f"Error processing query: {e}"

# def main():
#     parser = argparse.ArgumentParser(description="Interactive RAG pipeline for entertainment queries")
#     parser.add_argument("--query", help="Query for the RAG system")
#     parser.add_argument("--top_k", type=int, default=None, help=f"Number of chunks to retrieve (default: {config['retrieval']['top_k']})")
#     parser.add_argument("--max_new_tokens", type=int, default=None, help=f"Maximum new tokens for generation (default: {config['generation']['parameters']['max_new_tokens']})")
#     parser.add_argument("--temperature", type=float, default=None, help=f"Temperature for generation (default: {config['generation']['parameters']['temperature']})")
#     parser.add_argument("--top_k_sampling", type=int, default=None, help=f"Top-k for sampling (default: {config['generation']['parameters'].get('top_k', 50)})")
    
#     args = parser.parse_args()
    
#     logger.info("Starting RAG pipeline...")
#     try:
#         get_generator()
#         logger.info("Generator initialized successfully!")
#     except Exception as e:
#         logger.error(f"Failed to initialize generator: {e}")
#         return
    
#     if not args.query:
#         print("🎬 Entertainment RAG System Ready!")
#         print("Ask me about movies, anime, manga, K-dramas, or Bollywood!")
#         print("Type 'quit' to exit.\n")
        
#         while True:
#             try:
#                 query = input("🎭 Your query > ").strip()
#                 if query.lower() in ["quit", "exit", "q"]:
#                     print("Goodbye! 👋")
#                     break
                    
#                 if query:
#                     print(f"\n🔍 Processing: {query}")
#                     response = asyncio.run(run_rag(
#                         query=query,
#                         top_k=args.top_k,
#                         max_new_tokens=args.max_new_tokens,
#                         temperature=args.temperature,
#                         top_k_sampling=args.top_k_sampling
#                     ))
#                     print(f"\n🎯 Response:\n{response}\n")
#                     print("-" * 60)
#                 else:
#                     print("Please enter a valid query.")
                    
#             except KeyboardInterrupt:
#                 print("\n\nGoodbye! 👋")
#                 break
#             except Exception as e:
#                 logger.error(f"Error in interactive mode: {e}")
#                 print(f"Error: {e}")
#     else:
#         print(f"🔍 Processing query: {args.query}")
#         response = asyncio.run(run_rag(
#             query=args.query,
#             top_k=args.top_k,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_k_sampling=args.top_k_sampling
#         ))
#         print(f"\n🎯 Response:\n{response}")

# if __name__ == "__main__":
#     main()






import sys
from pathlib import Path
import yaml
import logging
import json
import argparse
import asyncio
from datetime import datetime
# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))



from retrieval.retriever import retrieve
from generation.generator import Generator
from memory.conversation_memory import ConversationMemory
from scripts.reddit_scraper import SmartRedditScraperBotWithEmbeddings



# Load config
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Logging setup
log_dir = BASE_DIR / config["pipeline"]["logging"]["log_dir"]
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / config["pipeline"]["logging"]["log_file"]
reddit_log_path = log_dir / config["pipeline"]["logging"]["reddit_log_file"]
logging.basicConfig(
    level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize generator and memory
generator = None
memory = ConversationMemory(max_turns=config["generation"].get("history_window", 10))

def detect_query_type(query):
    """Detect query type: factual, opinion, or recommendation"""
    query_lower = query.lower()
    factual_keywords = ["plot", "story", "ending", "summary", "explain", "what happens", "details"]
    opinion_keywords = ["think", "feel", "opinion", "view", "believe", "discussion", "users say"]
    recommendation_keywords = ["recommend", "suggest", "similar to", "like", "best", "top"]
    
    if any(keyword in query_lower for keyword in recommendation_keywords):
        return "recommendation"
    elif any(keyword in query_lower for keyword in opinion_keywords):
        return "opinion"
    elif any(keyword in query_lower for keyword in factual_keywords):
        return "factual"
    return "factual"  # Default to factual

async def retrieve_vector(query, top_k):
    """Retrieve vector store chunks asynchronously"""
    logger.info(f"Retrieving top {top_k} chunks from vector store...")
    chunks = await retrieve(query, top_k)  # retriever.py is async-native
    logger.info(f"Retrieved {len(chunks)} vector store chunks")
    return chunks

async def retrieve_reddit(query):
    """Retrieve Reddit data asynchronously"""
    logger.info("Triggering Reddit scraper...")
    scraper = SmartRedditScraperBotWithEmbeddings()
    result = await asyncio.to_thread(scraper.search_and_preprocess_with_embeddings, query, num_posts=config["reddit"]["scraper"]["max_posts"])
    reddit_data = result.get('data', [])
    logger.info(f"Retrieved {len(reddit_data)} Reddit items")
    with open(reddit_log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} - INFO - Retrieved {len(reddit_data)} Reddit items for query: {query}\n")
    return reddit_data

def get_generator():
    """Get or create the global generator instance"""
    global generator
    if generator is None:
        logger.info("Initializing Generator (first time)...")
        generator = Generator()
    return generator

async def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
    """Run the RAG pipeline with Reddit integration"""
    logger.info(f"Processing query: {query}")
    
    # Use config defaults
    top_k = top_k or config["retrieval"]["top_k"]
    max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
    temperature = temperature or config["generation"]["parameters"]["temperature"]
    top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
    try:
        # Detect query type and Reddit trigger
        query_type = detect_query_type(query)
        use_reddit = any(trigger in query.lower() for trigger in config["reddit"]["trigger_keywords"])
        logger.info(f"Query type: {query_type}, Use Reddit: {use_reddit}")
        
        # Parallel retrieval
        vector_chunks = []
        reddit_data = []
        if use_reddit:
            tasks = [
                retrieve_vector(query, top_k),
                retrieve_reddit(query)
            ]
            vector_chunks, reddit_data = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            vector_chunks = await retrieve_vector(query, top_k)
        
        # Handle potential errors
        if isinstance(vector_chunks, Exception):
            logger.error(f"Vector retrieval failed: {vector_chunks}")
            vector_chunks = []
        if isinstance(reddit_data, Exception):
            logger.error(f"Reddit retrieval failed: {reddit_data}")
            with open(reddit_log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} - ERROR - Reddit retrieval failed: {reddit_data}\n")
            reddit_data = []
        
        # Check vector store similarity
        use_only_reddit = False
        if vector_chunks and not reddit_data:
            max_similarity = max(chunk['similarity'] for chunk in vector_chunks)
            if max_similarity < config["retrieval"]["similarity_threshold"]:
                logger.info(f"Vector store similarity {max_similarity:.4f} < {config['retrieval']['similarity_threshold']}, but no Reddit data available")
        
        if vector_chunks and reddit_data:
            max_similarity = max(chunk['similarity'] for chunk in vector_chunks)
            if max_similarity < config["retrieval"]["similarity_threshold"]:
                logger.info(f"Vector store similarity {max_similarity:.4f} < {config['retrieval']['similarity_threshold']}, using Reddit data only")
                use_only_reddit = True
        
        # Prepare context
        vector_context = [chunk['text'] for chunk in vector_chunks] if not use_only_reddit else []
        reddit_context = [item['text'] for item in reddit_data] if reddit_data else []
        
        # Generate response
        gen = get_generator()
        logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
        response = gen.generate(
            query=query,
            vector_context=vector_context,
            reddit_context=reddit_context,
            memory=memory,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling
        )
        
        # Store conversation turn
        memory.add_turn(query, response)
        
        # Save result
        result = {
            "query": query,
            "response": response,
            "vector_chunks": vector_chunks,
            "reddit_data": reddit_data,
            "query_type": query_type,
            "parameters": {
                "top_k_retrieval": top_k,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k_sampling": top_k_sampling,
                "used_reddit": use_reddit,
                "used_only_reddit": use_only_reddit
            }
        }
        
        output_dir = BASE_DIR / config["pipeline"]["output"]["results_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_query = "".join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        output_file = output_dir / f"result_{safe_query}.json"
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result to {output_file}")
        
        return response
        
    except Exception as e:
        logger.exception(f"RAG pipeline failed: {e}")
        with open(reddit_log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} - ERROR - RAG pipeline failed: {e}\n")
        return f"Error processing query: {e}"

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG pipeline for entertainment queries")
    parser.add_argument("--query", help="Query for the RAG system")
    parser.add_argument("--top_k", type=int, default=None, help=f"Number of chunks to retrieve (default: {config['retrieval']['top_k']})")
    parser.add_argument("--max_new_tokens", type=int, default=None, help=f"Maximum new tokens for generation (default: {config['generation']['parameters']['max_new_tokens']})")
    parser.add_argument("--temperature", type=float, default=None, help=f"Temperature for generation (default: {config['generation']['parameters']['temperature']})")
    parser.add_argument("--top_k_sampling", type=int, default=None, help=f"Top-k for sampling (default: {config['generation']['parameters'].get('top_k', 50)})")
    
    args = parser.parse_args()
    
    logger.info("Starting RAG pipeline...")
    try:
        get_generator()
        logger.info("Generator initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return
    
    if not args.query:
        print("🎬 Entertainment RAG System Ready!")
        print("Ask me about movies, anime, manga, K-dramas, or Bollywood!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                query = input("🎭 Your query > ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye! 👋")
                    break
                    
                if query:
                    print(f"\n🔍 Processing: {query}")
                    response = asyncio.run(run_rag(
                        query=query,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k_sampling=args.top_k_sampling
                    ))
                    print(f"\n🎯 Response:\n{response}\n")
                    print("-" * 60)
                else:
                    print("Please enter a valid query.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")
    else:
        print(f"🔍 Processing query: {args.query}")
        response = asyncio.run(run_rag(
            query=args.query,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k_sampling=args.top_k_sampling
        ))
        print(f"\n🎯 Response:\n{response}")

if __name__ == "__main__":
    main()