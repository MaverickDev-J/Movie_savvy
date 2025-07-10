import sys
from pathlib import Path
import yaml
import logging 
import json
import argparse
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retriever import retrieve
from generation.generator import Generator
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

# Initialize generator
generator = None

# FastAPI app
app = FastAPI(title="Entertainment RAG API", description="RAG pipeline for entertainment queries")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "https://*.cloudspaces.litng.ai",  # Allow Lightning AI cloudspaces
        "*"  # Allow all origins for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k_sampling: Optional[int] = None

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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling
        )
        
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
        
        return result
        
    except Exception as e:
        logger.exception(f"RAG pipeline failed: {e}")
        with open(reddit_log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} - ERROR - RAG pipeline failed: {e}\n")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize generator on startup"""
    logger.info("Starting RAG pipeline...")
    try:
        get_generator()
        logger.info("Generator initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {e}")

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query using the RAG pipeline"""
    try:
        result = await run_rag(
            query=request.query,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k_sampling=request.top_k_sampling
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query")
async def process_query_get(query: str, top_k: Optional[int] = None, max_new_tokens: Optional[int] = None, 
                          temperature: Optional[float] = None, top_k_sampling: Optional[int] = None):
    """Process a query using the RAG pipeline (GET request)"""
    try:
        result = await run_rag(
            query=query,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k_sampling=top_k_sampling
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG API is running"}

def main():
    """Command line interface for interactive mode"""
    parser = argparse.ArgumentParser(description="Interactive RAG pipeline for entertainment queries")
    parser.add_argument("--query", help="Query for the RAG system")
    parser.add_argument("--top_k", type=int, default=None, help=f"Number of chunks to retrieve (default: {config['retrieval']['top_k']})")
    parser.add_argument("--max_new_tokens", type=int, default=None, help=f"Maximum new tokens for generation (default: {config['generation']['parameters']['max_new_tokens']})")
    parser.add_argument("--temperature", type=float, default=None, help=f"Temperature for generation (default: {config['generation']['parameters']['temperature']})")
    parser.add_argument("--top_k_sampling", type=int, default=None, help=f"Top-k for sampling (default: {config['generation']['parameters'].get('top_k', 50)})")
    parser.add_argument("--server", action="store_true", help="Run FastAPI server instead of CLI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server to")
    
    args = parser.parse_args()
    
    if args.server:
        # Run FastAPI server
        import uvicorn
        logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Run CLI mode
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
                        result = asyncio.run(run_rag(
                            query=query,
                            top_k=args.top_k,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_k_sampling=args.top_k_sampling
                        ))
                        response = result.get('response', result) if isinstance(result, dict) else result
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
            result = asyncio.run(run_rag(
                query=args.query,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k_sampling=args.top_k_sampling
            ))
            response = result.get('response', result) if isinstance(result, dict) else result
            print(f"\n🎯 Response:\n{response}")

if __name__ == "__main__":
    main()
