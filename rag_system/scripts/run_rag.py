# # base ------------------------------------------------------


# import sys
# from pathlib import Path
# import yaml
# import logging
# import json
# import argparse
# import asyncio
# from datetime import datetime

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# from retrieval.retriever import retrieve
# from generation.generator import Generator
# from scripts.reddit_scraper import SmartRedditScraperBotWithEmbeddings

# # Load config
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Logging setup
# log_dir = BASE_DIR / config["pipeline"]["logging"]["log_dir"]
# log_dir.mkdir(parents=True, exist_ok=True)
# log_path = log_dir / config["pipeline"]["logging"]["log_file"]
# reddit_log_path = log_dir / config["pipeline"]["logging"]["reddit_log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize generator
# generator = None

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
#     chunks = await retrieve(query, top_k)  # retriever.py is async-native
#     logger.info(f"Retrieved {len(chunks)} vector store chunks")
#     return chunks

# async def retrieve_reddit(query):
#     """Retrieve Reddit data asynchronously"""
#     logger.info("Triggering Reddit scraper...")
#     scraper = SmartRedditScraperBotWithEmbeddings()
#     result = await asyncio.to_thread(scraper.search_and_preprocess_with_embeddings, query, num_posts=config["reddit"]["scraper"]["max_posts"])
#     reddit_data = result.get('data', [])
#     logger.info(f"Retrieved {len(reddit_data)} Reddit items")
#     with open(reddit_log_path, "a") as f:
#         f.write(f"{datetime.now().isoformat()} - INFO - Retrieved {len(reddit_data)} Reddit items for query: {query}\n")
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
#         use_reddit = any(trigger in query.lower() for trigger in config["reddit"]["trigger_keywords"])
#         logger.info(f"Query type: {query_type}, Use Reddit: {use_reddit}")
        
#         # Parallel retrieval
#         vector_chunks = []
#         reddit_data = []
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
#             with open(reddit_log_path, "a") as f:
#                 f.write(f"{datetime.now().isoformat()} - ERROR - Reddit retrieval failed: {reddit_data}\n")
#             reddit_data = []
        
#         # Check vector store similarity
#         use_only_reddit = False
#         if vector_chunks and not reddit_data:
#             max_similarity = max(chunk['similarity'] for chunk in vector_chunks)
#             if max_similarity < config["retrieval"]["similarity_threshold"]:
#                 logger.info(f"Vector store similarity {max_similarity:.4f} < {config['retrieval']['similarity_threshold']}, but no Reddit data available")
        
#         if vector_chunks and reddit_data:
#             max_similarity = max(chunk['similarity'] for chunk in vector_chunks)
#             if max_similarity < config["retrieval"]["similarity_threshold"]:
#                 logger.info(f"Vector store similarity {max_similarity:.4f} < {config['retrieval']['similarity_threshold']}, using Reddit data only")
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
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k_sampling
#         )
        
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
#         with open(reddit_log_path, "a") as f:
#             f.write(f"{datetime.now().isoformat()} - ERROR - RAG pipeline failed: {e}\n")
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









# import sys
# from pathlib import Path
# import yaml
# import logging
# import json
# import argparse
# import asyncio
# from datetime import datetime

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# from retrieval.retriever import retrieve
# from generation.generator import Generator

# # Load config
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Logging setup
# log_dir = BASE_DIR / config["pipeline"]["logging"]["log_dir"]
# log_dir.mkdir(parents=True, exist_ok=True)
# log_path = log_dir / config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize generator
# generator = None

# def detect_query_type(query):
#     """Detect query type: factual, opinion, or recommendation"""
#     query_lower = query.lower()
#     factual_keywords = ["plot", "story", "ending", "summary", "explain", "what happens", "details"]
#     opinion_keywords = ["think", "feel", "opinion", "view", "believe"]
#     recommendation_keywords = ["recommend", "suggest", "similar to", "like", "best", "top"]
    
#     if any(keyword in query_lower for keyword in recommendation_keywords):
#         return "recommendation"
#     elif any(keyword in query_lower for keyword in opinion_keywords):
#         return "opinion"
#     return "factual"  # Default to factual

# async def retrieve_vector(query, top_k):
#     """Retrieve vector store chunks asynchronously"""
#     logger.info(f"Retrieving top {top_k} chunks from vector store...")
#     chunks = await retrieve(query, top_k)  # retriever.py is async-native
#     logger.info(f"Retrieved {len(chunks)} vector store chunks")
#     return chunks

# def get_generator():
#     """Get or create the global generator instance"""
#     global generator
#     if generator is None:
#         logger.info("Initializing Generator (first time)...")
#         generator = Generator()
#     return generator

# async def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
#     """Run the RAG pipeline"""
#     logger.info(f"Processing query: {query}")
    
#     # Use config defaults
#     top_k = top_k or config["retrieval"]["top_k"]
#     max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
#     temperature = temperature or config["generation"]["parameters"]["temperature"]
#     top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
#     try:
#         # Detect query type
#         query_type = detect_query_type(query)
#         logger.info(f"Query type: {query_type}")
        
#         # Retrieve vector chunks
#         vector_chunks = await retrieve_vector(query, top_k)
        
#         # Handle potential errors
#         if isinstance(vector_chunks, Exception):
#             logger.error(f"Vector retrieval failed: {vector_chunks}")
#             vector_chunks = []
        
#         # Check vector store similarity
#         vector_context = [chunk['text'] for chunk in vector_chunks]
#         if vector_chunks:
#             max_similarity = max(chunk['similarity'] for chunk in vector_chunks)
#             if max_similarity < config["retrieval"]["similarity_threshold"]:
#                 logger.info(f"Vector store similarity {max_similarity:.4f} < {config['retrieval']['similarity_threshold']}")
        
#         # Generate response
#         gen = get_generator()
#         logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
#         response = gen.generate(
#             query=query,
#             vector_context=vector_context,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k_sampling
#         )
        
#         # Save result
#         result = {
#             "query": query,
#             "response": response,
#             "vector_chunks": vector_chunks,
#             "query_type": query_type,
#             "parameters": {
#                 "top_k_retrieval": top_k,
#                 "max_new_tokens": max_new_tokens,
#                 "temperature": temperature,
#                 "top_k_sampling": top_k_sampling
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










# tavily 1


# import sys
# from pathlib import Path
# import yaml
# import logging
# import json
# import argparse
# import asyncio
# from datetime import datetime
# import os
# from dotenv import load_dotenv
# from tavily import TavilyClient

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# from retrieval.retriever import retrieve
# from generation.generator import Generator

# # Load config
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Load environment variables from .env file
# load_dotenv(dotenv_path=BASE_DIR / ".env")

# # Logging setup
# log_dir = BASE_DIR / config["pipeline"]["logging"]["log_dir"]
# log_dir.mkdir(parents=True, exist_ok=True)
# log_path = log_dir / config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_path),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize generator
# generator = None

# def detect_query_type(query):
#     """Detect query type: factual, opinion, or recommendation"""
#     query_lower = query.lower()
#     factual_keywords = ["plot", "story", "ending", "summary", "explain", "what happens", "details"]
#     opinion_keywords = ["think", "feel", "opinion", "view", "believe"]
#     recommendation_keywords = ["recommend", "suggest", "similar to", "like", "best", "top"]
    
#     if any(keyword in query_lower for keyword in recommendation_keywords):
#         return "recommendation"
#     elif any(keyword in query_lower for keyword in opinion_keywords):
#         return "opinion"
#     return "factual"  # Default to factual

# async def retrieve_vector(query, top_k):
#     """Retrieve vector store chunks asynchronously"""
#     logger.info(f"Retrieving top {top_k} chunks from vector store...")
#     chunks = await retrieve(query, top_k)  # retriever.py is async-native
#     logger.info(f"Retrieved {len(chunks)} vector store chunks")
#     return chunks

# def get_generator():
#     """Get or create the global generator instance"""
#     global generator
#     if generator is None:
#         logger.info("Initializing Generator (first time)...")
#         generator = Generator()
#     return generator

# async def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
#     """Run the RAG pipeline with Tavily integration"""
#     logger.info(f"Processing query: {query}")
    
#     # Use config defaults
#     top_k = top_k or config["retrieval"]["top_k"]
#     max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
#     temperature = temperature or config["generation"]["parameters"]["temperature"]
#     top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
#     try:
#         # Detect query type
#         query_type = detect_query_type(query)
#         logger.info(f"Query type: {query_type}")
        
#         # Retrieve vector chunks
#         vector_chunks = await retrieve_vector(query, top_k)
        
#         # Handle potential errors
#         if isinstance(vector_chunks, Exception):
#             logger.error(f"Vector retrieval failed: {vector_chunks}")
#             vector_chunks = []
        
#         # Prepare local context
#         vector_context = [chunk['text'] for chunk in vector_chunks]
        
#         # Define conditions for Tavily search
#         web_search_keywords = ["latest", "current", "news", "reviews", "upcoming", "new", "recent"]
#         use_tavily = (
#             len(vector_chunks) < 2 or
#             (vector_chunks and max(chunk['similarity'] for chunk in vector_chunks) < 0.6) or
#             any(keyword in query.lower() for keyword in web_search_keywords)
#         )
        
#         # Perform Tavily search if conditions are met
#         if use_tavily:
#             logger.info("Using Tavily for web search")
#             api_key = os.getenv("TAVILY_API_KEY")
#             if not api_key:
#                 logger.error("TAVILY_API_KEY not set in .env file")
#                 web_context = []
#             else:
#                 try:
#                     tavily_client = TavilyClient(api_key=api_key)
#                     response = tavily_client.search(
#                         query=query,
#                         search_depth="advanced",
#                         include_answer=True,
#                         include_raw_content=True,
#                         max_results=3
#                     )
#                     web_results = response.get("results", [])
#                     logger.info(f"Retrieved {len(web_results)} web results from Tavily")
#                     web_context = [result['content'] for result in web_results]  # Directly use content
#                 except Exception as e:
#                     logger.error(f"Tavily search failed: {e}")
#                     web_context = []
#         else:
#             logger.info("Using only local retrieval")
#             web_context = []
        
#         # Generate response
#         gen = get_generator()
#         logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
#         response = gen.generate(
#             query=query,
#             local_context=vector_context,
#             web_context=web_context,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k_sampling
#         )
        
#         # Save result
#         result = {
#             "query": query,
#             "response": response,
#             "vector_chunks": vector_chunks,
#             "web_context": web_context,
#             "query_type": query_type,
#             "parameters": {
#                 "top_k_retrieval": top_k,
#                 "max_new_tokens": max_new_tokens,
#                 "temperature": temperature,
#                 "top_k_sampling": top_k_sampling
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
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retriever import retrieve
from generation.generator import Generator

# Load config
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Logging setup
log_dir = BASE_DIR / config["pipeline"]["logging"]["log_dir"]
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / config["pipeline"]["logging"]["log_file"]
logging.basicConfig(
    level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize generator
generator = None

def analyze_query_intent(query):
    """Enhanced query analysis for web search intent and query type"""
    query_lower = query.lower()
    
    # Explicit web search indicators
    explicit_web_keywords = [
        "latest", "current", "recent", "today", "now", "2024", "2025",
        "search online", "web search", "internet", "google", "news",
        "update", "newest", "just released", "breaking", "trending"
    ]
    
    # Ongoing/current content indicators  
    current_content_keywords = [
        "ongoing", "current arc", "latest episode", "recent chapter",
        "new season", "upcoming", "this year", "this month"
    ]
    
    # Query type classification
    factual_keywords = ["plot", "story", "ending", "summary", "explain", "what happens", "details"]
    opinion_keywords = ["think", "feel", "opinion", "view", "believe", "theories", "fan theories"]
    recommendation_keywords = ["recommend", "suggest", "similar to", "like", "best", "top"]
    
    # Determine web search necessity
    explicit_web_intent = any(keyword in query_lower for keyword in explicit_web_keywords)
    current_content_query = any(keyword in query_lower for keyword in current_content_keywords)
    needs_web_search = explicit_web_intent or current_content_query
    
    # Determine query type
    if any(keyword in query_lower for keyword in recommendation_keywords):
        query_type = "recommendation"
    elif any(keyword in query_lower for keyword in opinion_keywords):
        query_type = "opinion" 
    else:
        query_type = "factual"
    
    return {
        "query_type": query_type,
        "needs_web_search": needs_web_search,
        "explicit_web_intent": explicit_web_intent,
        "current_content_query": current_content_query
    }

async def retrieve_vector(query, top_k):
    """Retrieve vector store chunks asynchronously"""
    logger.info(f"Retrieving top {top_k} chunks from vector store...")
    chunks = await retrieve(query, top_k)  # retriever.py is async-native
    logger.info(f"Retrieved {len(chunks)} vector store chunks")
    return chunks

def get_generator():
    """Get or create the global generator instance"""
    global generator
    if generator is None:
        logger.info("Initializing Generator (first time)...")
        generator = Generator()
    return generator

async def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None):
    """Run the RAG pipeline with Tavily integration"""
    logger.info(f"Processing query: {query}")
    
    # Use config defaults
    top_k = top_k or config["retrieval"]["top_k"]
    max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
    temperature = temperature or config["generation"]["parameters"]["temperature"]
    top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
    try:
        # Analyze query intent
        query_analysis = analyze_query_intent(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Retrieve vector chunks
        vector_chunks = await retrieve_vector(query, top_k)
        
        # Handle potential errors
        if isinstance(vector_chunks, Exception):
            logger.error(f"Vector retrieval failed: {vector_chunks}")
            vector_chunks = []
        
        # Prepare local context
        vector_context = [chunk['text'] for chunk in vector_chunks]
        
        # Enhanced web search decision
        use_tavily = (
            query_analysis["needs_web_search"] or  # Explicit or current content intent
            len(vector_chunks) < 2 or  # Insufficient local results
            (vector_chunks and max(chunk['similarity'] for chunk in vector_chunks) < 0.5) or  # Low similarity threshold
            (query_analysis["query_type"] == "opinion" and "theories" in query.lower())  # Fan theories need current info
        )

        logger.info(f"Web search decision: {use_tavily} (Reason: {'explicit_intent' if query_analysis['needs_web_search'] else 'fallback'})")
        
        # Enhanced Tavily search with better query construction
        if use_tavily:
            logger.info("Using Tavily for web search")
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.error("TAVILY_API_KEY not set in .env file")
                web_context = []
            else:
                try:
                    tavily_client = TavilyClient(api_key=api_key)
                    
                    # Construct better search query for Tavily
                    search_query = query
                    if query_analysis["current_content_query"]:
                        search_query = f"{query} 2024 2025 latest recent"
                    elif query_analysis["explicit_web_intent"]:
                        search_query = f"{query} current news recent"
                        
                    response = tavily_client.search(
                        query=search_query,
                        search_depth="advanced",
                        include_answer=True,
                        include_raw_content=True,
                        max_results=5 if query_analysis["needs_web_search"] else 3
                    )
                    web_results = response.get("results", [])
                    logger.info(f"Retrieved {len(web_results)} web results from Tavily")
                    
                    # Better content extraction
                    web_context = []
                    for result in web_results:
                        content = result.get('content', '')
                        if content and len(content) > 50:  # Filter out too short content
                            web_context.append(content[:1000])  # Limit length per result
                            
                except Exception as e:
                    logger.error(f"Tavily search failed: {e}")
                    web_context = []
        else:
            logger.info("Using only local retrieval")
            web_context = []
        
        # Generate response
        gen = get_generator()
        logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
        response = gen.generate(
            query=query,
            local_context=vector_context,
            web_context=web_context,
            query_analysis=query_analysis,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling
        )
        
        # Save result
        result = {
            "query": query,
            "response": response,
            "vector_chunks": vector_chunks,
            "web_context": web_context,
            "query_analysis": query_analysis,  # Add this line
            "search_strategy": "web_primary" if query_analysis["needs_web_search"] else "local_primary",  # Add this line
            "parameters": {
                "top_k_retrieval": top_k,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k_sampling": top_k_sampling
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