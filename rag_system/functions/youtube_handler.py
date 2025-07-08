import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List, Dict
from googleapiclient.discovery import build
import yaml
import requests
import re
from urllib.parse import urlparse, parse_qs

load_dotenv()

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "rag_system"
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"

# Logging setup
log_dir = BASE_DIR / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "rag_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class YouTubeTranscriptFetcher:
    def __init__(self, api_key):
        """
        Initialize the YouTube Transcript Fetcher
        
        Args:
            api_key (str): Your SearchAPI.io API key
        """
        self.api_key = api_key
        self.base_url = "https://www.searchapi.io/api/v1/search"
        
    def extract_video_id(self, youtube_url):
        """
        Extract video ID from various YouTube URL formats
        
        Args:
            youtube_url (str): YouTube video URL
            
        Returns:
            str: Video ID or None if not found
        """
        # Common YouTube URL patterns
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        # Try parsing as URL
        try:
            parsed_url = urlparse(youtube_url)
            if 'youtube.com' in parsed_url.netloc:
                query_params = parse_qs(parsed_url.query)
                if 'v' in query_params:
                    return query_params['v'][0]
            elif 'youtu.be' in parsed_url.netloc:
                return parsed_url.path[1:]  # Remove leading slash
        except:
            pass
            
        return None
    
    def get_transcript(self, youtube_url, lang='en'):
        """
        Get transcript for a YouTube video
        
        Args:
            youtube_url (str): YouTube video URL
            lang (str): Language code (default: 'en')
            
        Returns:
            dict: Response containing transcript data or error
        """
        # Extract video ID
        video_id = self.extract_video_id(youtube_url)
        if not video_id:
            return {
                'success': False,
                'error': 'Could not extract video ID from URL'
            }
        
        # Prepare API request
        params = {
            'engine': 'youtube_transcripts',
            'video_id': video_id,
            'lang': lang,
            'api_key': self.api_key
        }
        
        try:
            # Make API request
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'video_id': video_id,
                    'data': data
                }
            else:
                return {
                    'success': False,
                    'error': f'API request failed with status {response.status_code}',
                    'response': response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Request failed: {str(e)}'
            }

class YouTubeHandler:
    """Class to handle YouTube API interactions and transcript processing."""
    
    def __init__(self):
        """Initialize YouTube API client and transcript fetcher."""
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            youtube_config = config.get('youtube', {})
            
            # Load API keys from environment variables
            search_api_key = os.getenv('YOUTUBE_API_KEY')
            transcript_api_key = os.getenv('SEARCHAPI_TRANSCRIPT_KEY')
            
            if not search_api_key or not transcript_api_key:
                raise ValueError("Missing API keys in environment variables")
                
            self.youtube = build('youtube', 'v3', developerKey=search_api_key)
            self.transcript_fetcher = YouTubeTranscriptFetcher(transcript_api_key)
            self.max_results = youtube_config.get('max_results', 3)
            logger.info("YouTubeHandler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTubeHandler: {e}")
            raise
    
    def search_videos(self, query: str) -> List[Dict]:
        """Search for YouTube videos based on the query."""
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided to YouTube search")
                return []
                
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=self.max_results
            )
            response = request.execute()
            
            if 'items' not in response or not response['items']:
                logger.warning(f"No videos found for query: {query}")
                return []
                
            videos = [
                {
                    "id": item['id']['videoId'],
                    "title": item['snippet']['title'],
                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                for item in response['items']
            ]
            logger.info(f"Found {len(videos)} videos for query: {query}")
            return videos
        except Exception as e:
            logger.error(f"YouTube search failed for query '{query}': {e}")
            return []
    
    def get_transcript(self, video_id: str) -> List[Dict]:
        """Extract transcript for a given YouTube video ID using SearchAPI.io."""
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            result = self.transcript_fetcher.get_transcript(url)
            if result['success']:
                transcripts = result['data'].get('transcripts', [])
                # Rename 'start' to 'timestamp' for consistency
                for entry in transcripts:
                    if 'start' in entry:
                        entry['timestamp'] = entry.pop('start')
                logger.info(f"Extracted transcript for video ID: {video_id}")
                return transcripts
            else:
                logger.warning(f"Failed to extract transcript for video ID {video_id}: {result['error']}")
                return []
        except Exception as e:
            logger.error(f"Transcript fetching failed for video ID {video_id}: {e}")
            return []
    
    def chunk_transcript(self, transcript: List[Dict], chunk_size: int = 500) -> List[Dict]:
        """Chunk the transcript into smaller segments."""
        if not transcript:
            return []
        chunks = []
        current_chunk = ""
        for entry in transcript:
            text = entry.get('text', '')
            timestamp = entry.get('timestamp', 0)
            if len(current_chunk) + len(text) > chunk_size:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip(), "timestamp": timestamp})
                current_chunk = text
            else:
                current_chunk += " " + text
        if current_chunk:
            chunks.append({"text": current_chunk.strip(), "timestamp": transcript[-1].get('timestamp', 0)})
        logger.info(f"Chunked transcript into {len(chunks)} chunks")
        return chunks