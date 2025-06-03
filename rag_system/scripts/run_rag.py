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











import sys
from pathlib import Path

# Setup paths first
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import yaml
import logging
import json
import argparse
from retrieval.retriever import retrieve
from generation.generator import Generator

# Load config
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Logging setup
log_path = BASE_DIR / config["pipeline"]["logging"]["log_dir"] / config["pipeline"]["logging"]["log_file"]
logging.basicConfig(
    level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize generator once globally to avoid reloading model for each query
generator = None

def get_generator():
    """Get or create the global generator instance"""
    global generator
    if generator is None:
        logger.info("Initializing Generator (first time)...")
        generator = Generator()
    return generator

def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
    """Run the RAG pipeline with proper parameter handling"""
    logger.info(f"Processing query: {query}")
    
    # Use config defaults if parameters not provided
    top_k = top_k or config["retrieval"]["top_k"]
    max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
    temperature = temperature or config["generation"]["parameters"]["temperature"]
    top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
    try:
        # Retrieve chunks
        logger.info(f"Retrieving top {top_k} chunks...")
        chunks = retrieve(query, top_k)
        logger.info(f"Retrieved {len(chunks)} chunks")
        
        # Generate response using the global generator
        gen = get_generator()
        logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
        response = gen.generate(
            query=query,
            retrieved_chunks=chunks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling
        )
        
        # Save result
        result = {
            "query": query,
            "response": response,
            "retrieved_chunks": chunks,
            "parameters": {
                "top_k_retrieval": top_k,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k_sampling": top_k_sampling
            }
        }
        
        output_dir = BASE_DIR / config["pipeline"]["output"]["results_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a safe filename
        safe_query = "".join(c for c in query[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        output_file = output_dir / f"result_{safe_query}.json"
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result to {output_file}")
        
        return response
        
    except Exception as e:
        logger.exception(f"RAG pipeline failed: {e}")  # Use exception() to get full traceback
        return f"Error processing query: {e}"

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG pipeline for entertainment queries")
    parser.add_argument("--query", help="Query for the RAG system")
    parser.add_argument("--top_k", type=int, default=None, help=f"Number of chunks to retrieve (default: {config['retrieval']['top_k']})")
    parser.add_argument("--max_new_tokens", type=int, default=None, help=f"Maximum new tokens for generation (default: {config['generation']['parameters']['max_new_tokens']})")
    parser.add_argument("--temperature", type=float, default=None, help=f"Temperature for generation (default: {config['generation']['parameters']['temperature']})")
    parser.add_argument("--top_k_sampling", type=int, default=None, help=f"Top-k for sampling (default: {config['generation']['parameters'].get('top_k', 50)})")
    
    args = parser.parse_args()
    
    # Initialize generator once at startup
    logger.info("Starting RAG pipeline...")
    try:
        get_generator()  # This will initialize it
        logger.info("Generator initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return
    
    # Interactive mode if no query provided
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
                    response = run_rag(
                        query=query,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k_sampling=args.top_k_sampling
                    )
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
        # Single query mode
        print(f"🔍 Processing query: {args.query}")
        response = run_rag(
            query=args.query,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k_sampling=args.top_k_sampling
        )
        print(f"\n🎯 Response:\n{response}")

if __name__ == "__main__":
    main()