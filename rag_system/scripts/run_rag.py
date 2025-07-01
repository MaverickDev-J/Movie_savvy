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

# def analyze_query_intent(query):
#     """Enhanced query analysis for web search intent and query type"""
#     query_lower = query.lower()
    
#     # Explicit web search indicators
#     explicit_web_keywords = [
#         "latest", "current", "recent", "today", "now", "2024", "2025",
#         "search online", "web search", "internet", "google", "news",
#         "update", "newest", "just released", "breaking", "trending"
#     ]
    
#     # Ongoing/current content indicators  
#     current_content_keywords = [
#         "ongoing", "current arc", "latest episode", "recent chapter",
#         "new season", "upcoming", "this year", "this month"
#     ]
    
#     # Query type classification
#     factual_keywords = ["plot", "story", "ending", "summary", "explain", "what happens", "details"]
#     opinion_keywords = ["think", "feel", "opinion", "view", "believe", "theories", "fan theories"]
#     recommendation_keywords = ["recommend", "suggest", "similar to", "like", "best", "top"]
    
#     # Determine web search necessity
#     explicit_web_intent = any(keyword in query_lower for keyword in explicit_web_keywords)
#     current_content_query = any(keyword in query_lower for keyword in current_content_keywords)
#     needs_web_search = explicit_web_intent or current_content_query
    
#     # Determine query type
#     if any(keyword in query_lower for keyword in recommendation_keywords):
#         query_type = "recommendation"
#     elif any(keyword in query_lower for keyword in opinion_keywords):
#         query_type = "opinion" 
#     else:
#         query_type = "factual"
    
#     return {
#         "query_type": query_type,
#         "needs_web_search": needs_web_search,
#         "explicit_web_intent": explicit_web_intent,
#         "current_content_query": current_content_query
#     }

# async def get_web_results_async(query, query_analysis):
#     """Async web search function"""
#     api_key = os.getenv("TAVILY_API_KEY")
#     if not api_key:
#         logger.error("TAVILY_API_KEY not set in .env file")
#         return []
    
#     try:
#         def sync_web_search():
#             tavily_client = TavilyClient(api_key=api_key)
            
#             # Construct better search query for Tavily
#             search_query = query
#             if query_analysis["current_content_query"]:
#                 search_query = f"{query} 2024 2025 latest recent"
#             elif query_analysis["explicit_web_intent"]:
#                 search_query = f"{query} current news recent"
                
#             response = tavily_client.search(
#                 query=search_query,
#                 search_depth="advanced",
#                 include_answer=True,
#                 include_raw_content=True,
#                 max_results=5 if query_analysis["needs_web_search"] else 3
#             )
#             return response.get("results", [])
        
#         # Run sync function in thread pool
#         web_results = await asyncio.to_thread(sync_web_search)
#         logger.info(f"Retrieved {len(web_results)} web results from Tavily")
        
#         # Better content extraction
#         web_context = []
#         for result in web_results:
#             content = result.get('content', '')
#             if content and len(content) > 50:  # Filter out too short content
#                 web_context.append(content[:1000])  # Limit length per result
        
#         return web_context
        
#     except Exception as e:
#         logger.error(f"Tavily search failed: {e}")
#         return []

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
#     """Run the RAG pipeline with parallel processing"""
#     logger.info(f"Processing query: {query}")
    
#     # Use config defaults
#     top_k = top_k or config["retrieval"]["top_k"]
#     max_new_tokens = max_new_tokens or config["generation"]["parameters"]["max_new_tokens"]
#     temperature = temperature or config["generation"]["parameters"]["temperature"]
#     top_k_sampling = top_k_sampling or config["generation"]["parameters"].get("top_k", 50)
    
#     try:
#         # Analyze query intent
#         query_analysis = analyze_query_intent(query)
#         logger.info(f"Query analysis: {query_analysis}")
        
#         # Start both tasks in parallel
#         vector_task = asyncio.create_task(retrieve_vector(query, top_k))
        
#         # Start web search task if potentially needed
#         web_task = None
#         if query_analysis["needs_web_search"]:
#             web_task = asyncio.create_task(get_web_results_async(query, query_analysis))
        
#         # Wait for vector results first
#         vector_chunks = await vector_task
        
#         # Handle potential errors
#         if isinstance(vector_chunks, Exception):
#             logger.error(f"Vector retrieval failed: {vector_chunks}")
#             vector_chunks = []
        
#         # Prepare local context
#         vector_context = [chunk['text'] for chunk in vector_chunks]
        
#         # Decide if we need web results based on vector quality
#         need_web = (
#             query_analysis["needs_web_search"] or  # Explicit intent
#             len(vector_chunks) < 2 or  # Insufficient local results
#             (vector_chunks and max(chunk['similarity'] for chunk in vector_chunks) < 0.5)  # Low similarity
#         )
        
#         # Handle web results
#         web_context = []
#         if need_web:
#             if web_task is None:  # Start web search if not already started
#                 web_task = asyncio.create_task(get_web_results_async(query, query_analysis))
#             web_context = await web_task
#         elif web_task:
#             web_task.cancel()  # Cancel if not needed
#             logger.info("Web search cancelled - sufficient local results found")
        
#         logger.info(f"Web search decision: {need_web} (Reason: {'explicit_intent' if query_analysis['needs_web_search'] else 'quality_based'})")
        
#         # Generate response
#         gen = get_generator()
#         logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
#         response = gen.generate(
#             query=query,
#             local_context=vector_context,
#             web_context=web_context,
#             query_analysis=query_analysis,
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
#             "query_analysis": query_analysis,
#             "search_strategy": "web_primary" if query_analysis["needs_web_search"] else "local_primary",
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
    
    # Query type classification with enhanced logic
    factual_keywords = ["plot", "story", "ending", "summary", "explain", "what happens", "details"]
    opinion_keywords = ["think", "feel", "opinion", "view", "believe", "theories", "fan theories"]
    recommendation_keywords = ["recommend", "suggest", "similar to", "like", "best", "top"]
    current_keywords = ["latest", "recent", "current", "new", "today", "now"]
    
    # Determine web search necessity
    explicit_web_intent = any(keyword in query_lower for keyword in explicit_web_keywords)
    current_content_query = any(keyword in query_lower for keyword in current_content_keywords)
    needs_web_search = explicit_web_intent or current_content_query
    
    # Enhanced query type classification with priority handling
    query_types = []
    if any(keyword in query_lower for keyword in current_keywords):
        query_types.append('current')
    if any(keyword in query_lower for keyword in recommendation_keywords):
        query_types.append('recommendation')
    if any(keyword in query_lower for keyword in opinion_keywords):
        query_types.append('opinion')
    if any(keyword in query_lower for keyword in factual_keywords):
        query_types.append('factual')

    # Prioritize current events, then recommendations, then opinions, then factual
    if 'current' in query_types:
        query_type = 'current'
    elif 'recommendation' in query_types:
        query_type = 'recommendation'
    elif 'opinion' in query_types:
        query_type = 'opinion'
    elif 'factual' in query_types:
        query_type = 'factual'
    else:
        query_type = 'factual'  # default fallback
    
    return {
        "query_type": query_type,
        "needs_web_search": needs_web_search,
        "explicit_web_intent": explicit_web_intent,
        "current_content_query": current_content_query
    }

async def get_web_results_async(query, query_analysis):
    """Async web search function"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY not set in .env file")
        return []
    
    try:
        def sync_web_search():
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
            return response.get("results", [])
        
        # Run sync function in thread pool
        web_results = await asyncio.to_thread(sync_web_search)
        logger.info(f"Retrieved {len(web_results)} web results from Tavily")
        
        # Better content extraction
        web_context = []
        for result in web_results:
            content = result.get('content', '')
            if content and len(content) > 50:  # Filter out too short content
                web_context.append(content[:1000])  # Limit length per result
        
        return web_context
        
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

async def retrieve_vector(query, top_k, similarity_threshold=None):
    """Retrieve vector store chunks asynchronously with configurable similarity threshold"""
    logger.info(f"Retrieving top {top_k} chunks from vector store...")
    
    # Use default similarity threshold if not provided
    if similarity_threshold is None:
        similarity_threshold = config.get("retrieval", {}).get("similarity_threshold", 0.3)
    
    chunks = await retrieve(query, top_k)  # retriever.py is async-native
    
    # Filter chunks by similarity threshold if specified
    if similarity_threshold > 0:
        filtered_chunks = [chunk for chunk in chunks if chunk.get('similarity', 0) >= similarity_threshold]
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} using similarity threshold {similarity_threshold}")
        chunks = filtered_chunks
    
    logger.info(f"Retrieved {len(chunks)} vector store chunks")
    return chunks

def get_generator():
    """Get or create the global generator instance"""
    global generator
    if generator is None:
        logger.info("Initializing Generator (first time)...")
        generator = Generator()
    return generator

async def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None, similarity_threshold=None):
    """Run the RAG pipeline with parallel processing and configurable parameters for experiments"""
    logger.info(f"Processing query: {query}")
    
    # Use config defaults or provided parameters (important for experiments)
    top_k = top_k if top_k is not None else config["retrieval"]["top_k"]
    max_new_tokens = max_new_tokens if max_new_tokens is not None else config["generation"]["parameters"]["max_new_tokens"]
    temperature = temperature if temperature is not None else config["generation"]["parameters"]["temperature"]
    top_k_sampling = top_k_sampling if top_k_sampling is not None else config["generation"]["parameters"].get("top_k", 50)
    similarity_threshold = similarity_threshold if similarity_threshold is not None else config.get("retrieval", {}).get("similarity_threshold", 0.3)
    
    try:
        # Analyze query intent
        query_analysis = analyze_query_intent(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Start both tasks in parallel
        vector_task = asyncio.create_task(retrieve_vector(query, top_k, similarity_threshold))
        
        # Start web search task if potentially needed
        web_task = None
        if query_analysis["needs_web_search"]:
            web_task = asyncio.create_task(get_web_results_async(query, query_analysis))
        
        # Wait for vector results first
        vector_chunks = await vector_task
        
        # Handle potential errors
        if isinstance(vector_chunks, Exception):
            logger.error(f"Vector retrieval failed: {vector_chunks}")
            vector_chunks = []
        
        # Prepare local context
        vector_context = [chunk['text'] for chunk in vector_chunks]
        
        # Decide if we need web results based on vector quality
        need_web = (
            query_analysis["needs_web_search"] or  # Explicit intent
            len(vector_chunks) < 2 or  # Insufficient local results
            (vector_chunks and max(chunk.get('similarity', 0) for chunk in vector_chunks) < similarity_threshold)  # Low similarity
        )
        
        # Handle web results
        web_context = []
        if need_web:
            if web_task is None:  # Start web search if not already started
                web_task = asyncio.create_task(get_web_results_async(query, query_analysis))
            web_context = await web_task
        elif web_task:
            web_task.cancel()  # Cancel if not needed
            logger.info("Web search cancelled - sufficient local results found")
        
        logger.info(f"Web search decision: {need_web} (Reason: {'explicit_intent' if query_analysis['needs_web_search'] else 'quality_based'})")
        
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
            "query_analysis": query_analysis,
            "search_strategy": "web_primary" if query_analysis["needs_web_search"] else "local_primary",
            "parameters": {
                "top_k_retrieval": top_k,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k_sampling": top_k_sampling,
                "similarity_threshold": similarity_threshold
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
    parser.add_argument("--similarity_threshold", type=float, default=None, help=f"Similarity threshold for retrieval filtering (default: {config.get('retrieval', {}).get('similarity_threshold', 0.3)})")
    
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
                        top_k_sampling=args.top_k_sampling,
                        similarity_threshold=args.similarity_threshold
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
            top_k_sampling=args.top_k_sampling,
            similarity_threshold=args.similarity_threshold
        ))
        print(f"\n🎯 Response:\n{response}")

if __name__ == "__main__":
    main()