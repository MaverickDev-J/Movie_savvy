import sys
from pathlib import Path
import yaml
import logging
import json
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import dependencies
from retrieval.retriever import retrieve
from generation.generator import Generator
from rag_system.functions.function_manager import FunctionManager
from rag_system.functions.intent_classifier import EnhancedIntentClassifier
from rag_system.embeddings.semantic_scorer import SemanticContentProcessor
from rag_system.generation.response_refiner import ResponseRefiner

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

# Initialize global components
generator = None
function_manager = None

# FastAPI app
app = FastAPI(title="Enhanced Entertainment RAG API", description="Multi-source RAG pipeline for entertainment queries")

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
    similarity_threshold: Optional[float] = None

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

def get_function_manager():
    """Get or create the global function manager instance"""
    global function_manager
    if function_manager is None:
        logger.info("Initializing FunctionManager (first time)...")
        function_manager = FunctionManager()
    return function_manager

async def run_rag(query, top_k=None, max_new_tokens=None, temperature=None, top_k_sampling=None, similarity_threshold=None):
    """Run the RAG pipeline with parallel processing and configurable parameters"""
    logger.info(f"Processing query: {query}")
    
    # Use config defaults or provided parameters
    top_k = top_k if top_k is not None else config["retrieval"]["top_k"]
    max_new_tokens = max_new_tokens if max_new_tokens is not None else config["generation"]["parameters"]["max_new_tokens"]
    temperature = temperature if temperature is not None else config["generation"]["parameters"]["temperature"]
    top_k_sampling = top_k_sampling if top_k_sampling is not None else config["generation"]["parameters"].get("top_k", 50)
    similarity_threshold = similarity_threshold if similarity_threshold is not None else config.get("retrieval", {}).get("similarity_threshold", 0.3)
    
    try:
        # Initialize function manager
        fm = get_function_manager()
        
        # Use FunctionManager to acquire content from all sources
        logger.info("Acquiring content from multiple sources...")
        content_results = await fm.acquire_content(query)
        
        # Initialize semantic scorer
        semantic_scorer = SemanticContentProcessor()
        
        # Extract and score contexts
        vector_chunks = content_results.get("vector_results", [])
        web_context = content_results.get("web_results", [])
        youtube_context = []
        if content_results.get("youtube_results"):
            youtube_results = content_results["youtube_results"]
            for video_data in youtube_results:
                transcript_chunks = video_data.get("transcript_chunks", [])
                youtube_context.extend([chunk.get("text", "") for chunk in transcript_chunks if chunk.get("text", "")])
            logger.info(f"Retrieved {len(youtube_context)} YouTube transcript chunks")
        
        # Score and filter contexts
        vector_context = [chunk['text'] for chunk in semantic_scorer.score_content(query, vector_chunks, min_score=0.5)]
        web_context = semantic_scorer.score_content(query, web_context, min_score=0.5)
        youtube_context = semantic_scorer.score_content(query, youtube_context, min_score=0.5)
        
        # Fallback to original logic if FunctionManager doesn't provide sufficient results
        if not vector_chunks:
            logger.info("FunctionManager didn't return vector results, falling back to direct retrieval...")
            vector_task = asyncio.create_task(retrieve_vector(query, top_k, similarity_threshold))
            vector_chunks = await vector_task
        
        # Analyze query intent for generation
        query_analysis = analyze_query_intent(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Handle potential errors
        if isinstance(vector_chunks, Exception):
            logger.error(f"Vector retrieval failed: {vector_chunks}")
            vector_chunks = []
        
        # Prepare local context if not already prepared
        if not vector_context:
            vector_context = [chunk['text'] for chunk in vector_chunks if 'text' in chunk]
        
        # Decide if we need additional web results based on vector quality
        need_additional_web = (
            query_analysis["needs_web_search"] and not web_context or  # Explicit intent but no web results
            len(vector_chunks) < 2 or  # Insufficient local results
            (vector_chunks and max(chunk.get('similarity', 0) for chunk in vector_chunks) < similarity_threshold)  # Low similarity
        )
        
        # Get additional web results if needed and not already provided by FunctionManager
        if need_additional_web and not web_context:
            logger.info("Getting additional web results...")
            web_context = await get_web_results_async(query, query_analysis)
        
        logger.info(f"Final context summary - Vector: {len(vector_context)}, Web: {len(web_context)}, YouTube: {len(youtube_context)}")
        
        # Generate response
        gen = get_generator()
        logger.info(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k_sampling}")
        response = gen.generate(
            query=query,
            local_context=vector_context,
            web_context=web_context,
            youtube_context=youtube_context,
            query_analysis=query_analysis,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling
        )
        
        # Refine response
        refiner = ResponseRefiner()
        refined_response = refiner.refine_response(response, query, {
            "youtube": youtube_context,
            "web": web_context,
            "local": vector_context
        })
        
        # Save result
        result = {
            "query": query,
            "response": refined_response,
            "vector_chunks": vector_chunks,
            "web_context": web_context,
            "youtube_context": youtube_context,
            "query_analysis": query_analysis,
            "search_strategy": "multi_source",
            "content_sources_used": {
                "vector_store": len(vector_context) > 0,
                "web_search": len(web_context) > 0,
                "youtube": len(youtube_context) > 0
            },
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
        
        return result
        
    except Exception as e:
        logger.exception(f"RAG pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting Enhanced RAG pipeline...")
    try:
        get_generator()
        get_function_manager()
        logger.info("Generator and FunctionManager initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize components: {e}")

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query using the enhanced RAG pipeline"""
    try:
        result = await run_rag(
            query=request.query,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k_sampling=request.top_k_sampling,
            similarity_threshold=request.similarity_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query")
async def process_query_get(
    query: str, 
    top_k: Optional[int] = None, 
    max_new_tokens: Optional[int] = None, 
    temperature: Optional[float] = None, 
    top_k_sampling: Optional[int] = None,
    similarity_threshold: Optional[float] = None
):
    """Process a query using the enhanced RAG pipeline (GET request)"""
    try:
        result = await run_rag(
            query=query,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k_sampling=top_k_sampling,
            similarity_threshold=similarity_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Enhanced RAG API is running"}

# Status endpoint for debugging
@app.get("/status")
async def get_status():
    """Get system status and configuration"""
    return {
        "status": "running",
        "components": {
            "generator": generator is not None,
            "function_manager": function_manager is not None,
            "tavily_api_key": os.getenv("TAVILY_API_KEY") is not None
        },
        "config": {
            "retrieval_top_k": config.get("retrieval", {}).get("top_k", "not set"),
            "similarity_threshold": config.get("retrieval", {}).get("similarity_threshold", "not set"),
            "generation_params": config.get("generation", {}).get("parameters", {})
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)


    
