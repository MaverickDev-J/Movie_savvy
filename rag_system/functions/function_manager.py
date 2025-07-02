# import asyncio
# import logging
# from pathlib import Path
# from typing import Dict, List
# from rag_system.functions.intent_classifier import EnhancedIntentClassifier
# from rag_system.functions.youtube_handler import YouTubeHandler
# from rag_system.retrieval.retriever import retrieve

# # Setup paths
# BASE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_system"
# log_dir = BASE_DIR / "output" / "logs"
# log_dir.mkdir(parents=True, exist_ok=True)
# log_path = log_dir / "rag_pipeline.log"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# class FunctionManager:
#     """Orchestrates function calls for content acquisition."""
    
#     def __init__(self):
#         self.intent_classifier = EnhancedIntentClassifier()
#         self.youtube_handler = YouTubeHandler()
    
#     async def acquire_content(self, query: str) -> Dict:
#         """Acquire content from multiple sources in parallel."""
#         try:
#             intent_data = self.intent_classifier.analyze_query_intent(query)
#             tasks = [self.run_vector_search(query)]
#             if intent_data.get("needs_web_search", False):
#                 tasks.append(self.run_tavily_search(query))
#             if intent_data.get("needs_youtube", False):
#                 tasks.append(self.run_youtube_search(intent_data))
#             results = await asyncio.gather(*tasks, return_exceptions=True)
#             return {
#                 "vector_results": results[0],
#                 "web_results": results[1] if len(results) > 1 else None,
#                 "youtube_results": results[2] if len(results) > 2 else None
#             }
#         except Exception as e:
#             logger.error(f"Content acquisition failed: {e}")
#             return {}
    
#     async def run_vector_search(self, query: str) -> List[Dict]:
#         """Run vector search to retrieve relevant chunks."""
#         try:
#             return await retrieve(query)
#         except Exception as e:
#             logger.error(f"Vector search failed: {e}")
#             return []
    
#     async def run_tavily_search(self, query: str) -> List[str]:
#         """Run Tavily web search (placeholder)."""
#         # Placeholder: Implement Tavily search in Week 3 if needed
#         return []
    
#     async def run_youtube_search(self, intent_data: Dict) -> List[Dict]:
#         """Run YouTube search and process transcripts."""
#         try:
#             videos = self.youtube_handler.search_videos(intent_data["youtube_search_terms"][0])
#             processed_videos = []
#             for video in videos:
#                 transcript = self.youtube_handler.get_transcript(video["id"])
#                 if transcript:
#                     chunks = self.youtube_handler.chunk_transcript(transcript)
#                     processed_videos.append({
#                         "video_metadata": video,
#                         "transcript_chunks": chunks,
#                         "total_chunks": len(chunks)
#                     })
#             return processed_videos
#         except Exception as e:
#             logger.error(f"YouTube search failed: {e}")
#             return []










import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from rag_system.functions.intent_classifier import EnhancedIntentClassifier
from rag_system.functions.youtube_handler import YouTubeHandler
from rag_system.retrieval.retriever import retrieve

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_system"
log_dir = BASE_DIR / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "rag_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FunctionManager:
    """Orchestrates function calls for content acquisition."""
    
    def __init__(self):
        self.intent_classifier = EnhancedIntentClassifier()
        self.youtube_handler = YouTubeHandler()
    
    async def acquire_content(self, query: str) -> Dict:
        results = {"vector_results": [], "web_results": [], "youtube_results": []}
        
        try:
            intent_data = self.intent_classifier.analyze_query_intent(query)
            tasks = [self.run_vector_search(query)]
            if intent_data.get("needs_web_search", False):
                tasks.append(self.run_tavily_search(query))
            if intent_data.get("needs_youtube", False):
                youtube_videos = self.youtube_handler.search_videos(query)
                youtube_results = []
                for video in youtube_videos:
                    transcript = self.youtube_handler.get_transcript(video["id"])
                    chunks = self.youtube_handler.chunk_transcript(transcript)
                    youtube_results.append({
                        "id": video["id"],
                        "title": video["title"],
                        "url": video["url"],
                        "transcript_chunks": chunks  # Ensure chunks are passed
                    })
                results["youtube_results"] = youtube_results
            
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            results["vector_results"] = task_results[0]
            if len(task_results) > 1:
                results["web_results"] = task_results[1]
            
            return results
        except Exception as e:
            logger.error(f"Content acquisition failed: {e}")
            return results
    
    async def run_vector_search(self, query: str) -> List[Dict]:
        """Run vector search to retrieve relevant chunks."""
        try:
            return await retrieve(query)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def run_tavily_search(self, query: str) -> List[str]:
        """Run Tavily web search (placeholder)."""
        # Placeholder: Implement Tavily search in Week 3 if needed
        return []
    
    async def run_youtube_search(self, intent_data: Dict) -> List[Dict]:
        """Run YouTube search and process transcripts."""
        try:
            videos = self.youtube_handler.search_videos(intent_data["youtube_search_terms"][0])
            processed_videos = []
            for video in videos:
                transcript = self.youtube_handler.get_transcript(video["id"])
                if transcript:
                    chunks = self.youtube_handler.chunk_transcript(transcript)
                    processed_videos.append({
                        "video_metadata": video,
                        "transcript_chunks": chunks,
                        "total_chunks": len(chunks)
                    })
            return processed_videos
        except Exception as e:
            logger.error(f"YouTube search failed: {e}")
            return []