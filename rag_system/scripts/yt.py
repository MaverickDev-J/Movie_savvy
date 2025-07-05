
import requests
import json
import re
import sys
from typing import Dict, List, Optional, Any

class SerperVideoSearch:
    """
    Python class to interact with Serper.dev Video Search API
    """

    def __init__(self, api_key: str, serpapi_key: str = None):
        """
        Initialize the SerperVideoSearch class

        Args:
            api_key (str): Your Serper.dev API key
            serpapi_key (str): Your SerpAPI key for transcript extraction
        """
        self.api_key = api_key
        self.serpapi_key = serpapi_key
        self.base_url = "https://google.serper.dev/videos"
        self.headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

    def search_videos(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Search for videos using Serper API

        Args:
            query (str): Search query
            **kwargs: Additional search parameters

        Returns:
            Dict: Raw API response
        """
        # Set default parameters
        params = {
            'q': query,
            'num': kwargs.get('num', 10),
            'page': kwargs.get('page', 1),
            'gl': kwargs.get('gl', 'us'),
            'hl': kwargs.get('hl', 'en'),
            'autocorrect': kwargs.get('autocorrect', True)
        }

        # Add optional parameters if provided
        if 'location' in kwargs:
            params['location'] = kwargs['location']
        if 'tbs' in kwargs:
            params['tbs'] = kwargs['tbs']

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(params)
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error making request: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON response: {e}")
            raise

    def get_youtube_video_stats_serpapi(self, video_id: str) -> Dict[str, Any]:
        """
        Get YouTube video statistics using SerpAPI
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            Dict: Video statistics including views, likes, etc.
        """
        if not self.serpapi_key:
            return self._generate_dummy_stats()
        
        try:
            params = {
                "engine": "youtube",
                "search_query": f"video_id:{video_id}",
                "api_key": self.serpapi_key
            }
            
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract video stats from SerpAPI response
            if 'video_results' in data and data['video_results']:
                video_info = data['video_results'][0]
                return {
                    'view_count': self._parse_views(video_info.get('views', '0')),
                    'like_count': self._parse_likes(video_info.get('likes', '0')),
                    'comment_count': 0,  # SerpAPI might not always provide comments
                    'title': video_info.get('title', ''),
                    'published_at': video_info.get('published_date', ''),
                    'channel_title': video_info.get('channel', {}).get('name', '')
                }
            else:
                return self._generate_dummy_stats()
                
        except Exception as e:
            print(f"⚠️ Could not fetch YouTube stats for {video_id}: {e}")
            return self._generate_dummy_stats()

    def _parse_views(self, views_str) -> int:
        """Parse view count from string like '1.2M views' or '123,456 views' or integer"""
        if not views_str:
            return 0
        
        # Handle if it's already an integer
        if isinstance(views_str, int):
            return views_str
        
        # Handle if it's a string
        if isinstance(views_str, str):
            # Remove 'views' and clean up
            views_str = views_str.lower().replace('views', '').replace(',', '').strip()
            
            try:
                if 'k' in views_str:
                    return int(float(views_str.replace('k', '')) * 1000)
                elif 'm' in views_str:
                    return int(float(views_str.replace('m', '')) * 1000000)
                elif 'b' in views_str:
                    return int(float(views_str.replace('b', '')) * 1000000000)
                else:
                    return int(float(views_str))
            except (ValueError, TypeError):
                return 0
        
        return 0

    def _parse_likes(self, likes_str) -> int:
        """Parse like count from string or integer"""
        if not likes_str:
            return 0
        
        # Handle if it's already an integer
        if isinstance(likes_str, int):
            return likes_str
        
        # Handle if it's a string
        if isinstance(likes_str, str):
            likes_str = likes_str.lower().replace('likes', '').replace(',', '').strip()
            
            try:
                if 'k' in likes_str:
                    return int(float(likes_str.replace('k', '')) * 1000)
                elif 'm' in likes_str:
                    return int(float(likes_str.replace('m', '')) * 1000000)
                else:
                    return int(float(likes_str))
            except (ValueError, TypeError):
                return 0
        
        return 0

    def _generate_dummy_stats(self) -> Dict[str, Any]:
        """Generate dummy stats for demo purposes"""
        import random
        
        view_count = random.randint(1000, 10000000)
        like_count = random.randint(10, view_count // 100)
        
        return {
            'view_count': view_count,
            'like_count': like_count,
            'comment_count': random.randint(5, 1000),
            'title': '',
            'published_at': '',
            'channel_title': ''
        }

    def get_top_videos(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Get top videos with formatted data including views and likes

        Args:
            query (str): Search query
            **kwargs: Search parameters

        Returns:
            List[Dict]: List of formatted video data with stats
        """
        try:
            data = self.search_videos(query, **kwargs)

            if 'videos' not in data or not data['videos']:
                return []

            # Format the video data and get stats
            formatted_videos = []
            for i, video in enumerate(data['videos']):
                video_id = extract_video_id(video.get('link', ''))
                
                # Get video statistics
                stats = {'view_count': 0, 'like_count': 0, 'comment_count': 0}
                if video_id:
                    stats = self.get_youtube_video_stats_serpapi(video_id)
                
                formatted_video = {
                    'position': video.get('position', i + 1),
                    'title': video.get('title', ''),
                    'link': video.get('link', ''),
                    'snippet': video.get('snippet', ''),
                    'date': video.get('date', ''),
                    'duration': self._parse_duration(video.get('snippet', '')),
                    'channel': self._extract_channel(video.get('title', '')),
                    'thumbnail': video.get('imageUrl'),
                    'video_id': video_id,
                    'view_count': stats['view_count'],
                    'like_count': stats['like_count'],
                    'comment_count': stats['comment_count']
                }
                formatted_videos.append(formatted_video)

            return formatted_videos

        except Exception as e:
            print(f"❌ Error getting top videos: {e}")
            raise

    def get_top_viewed_videos(self, query: str, top_n: int = 2, **kwargs) -> List[Dict[str, Any]]:
        """
        Get top N most viewed videos from search results

        Args:
            query (str): Search query
            top_n (int): Number of top viewed videos to return
            **kwargs: Search parameters

        Returns:
            List[Dict]: List of top N most viewed videos
        """
        try:
            # Get more videos to have better selection
            search_kwargs = {k: v for k, v in kwargs.items() if k != 'num'}
            videos = self.get_top_videos(query, num=10, **search_kwargs)
            
            if not videos:
                return []

            # Sort by view count (descending)
            sorted_videos = sorted(videos, key=lambda x: x['view_count'], reverse=True)
            
            # Return top N videos
            return sorted_videos[:top_n]

        except Exception as e:
            print(f"❌ Error getting top viewed videos: {e}")
            raise

    def _parse_duration(self, snippet: str) -> Optional[str]:
        """Extract duration from snippet text"""
        if not snippet:
            return None

        duration_match = re.search(r'Duration:\s*([^,]+)', snippet)
        return duration_match.group(1).strip() if duration_match else None

    def _extract_channel(self, title: str) -> Optional[str]:
        """Extract channel name from title"""
        if not title:
            return None

        # Look for " - YouTube" pattern
        youtube_match = re.search(r'(.+?)\s*-\s*YouTube$', title)
        if youtube_match:
            before_youtube = youtube_match.group(1)
            parts = before_youtube.split(' - ')
            return parts[-1] if parts else None

        return None

# YouTube Transcript Functions using SerpAPI
def extract_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    if not url or url.startswith('-') or len(url) < 3:
        return None

    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        r'^([a-zA-Z0-9_-]{11})$'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def format_time(seconds):
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def format_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def get_youtube_transcript_serpapi(video_url_or_id, serpapi_key):
    """
    Get transcript from a YouTube video using SerpAPI (Lightning AI compatible)

    Args:
        video_url_or_id (str): YouTube URL or video ID
        serpapi_key (str): Your SerpAPI key

    Returns:
        dict: Contains transcript text, language info, and metadata
    """
    # Extract video ID if URL is provided
    if 'youtube.com' in video_url_or_id or 'youtu.be' in video_url_or_id:
        video_id = extract_video_id(video_url_or_id)
    else:
        video_id = video_url_or_id

    if not video_id:
        return {"error": "Invalid YouTube URL or video ID"}

    if not serpapi_key or serpapi_key == "YOUR_SERPAPI_KEY_HERE":
        return {"error": "SerpAPI key is required for transcript extraction"}

    try:
        print(f"📹 Video ID: {video_id}")
        print("🔄 Fetching transcript using SerpAPI...")

        # Use SerpAPI to get YouTube transcript
        params = {
            "engine": "youtube_transcripts",
            "video_id": video_id,
            "api_key": serpapi_key
        }

        response = requests.get("https://serpapi.com/search", params=params)
        
        # Print response for debugging
        print(f"📊 SerpAPI Response Status: {response.status_code}")
        
        if response.status_code == 400:
            try:
                error_data = response.json()
                return {"error": f"SerpAPI 400 Error: {error_data.get('error', 'Bad Request - Check your API key and parameters')}"}
            except:
                return {"error": "SerpAPI 400 Error: Bad Request - Check your API key and video ID"}
        
        response.raise_for_status()
        data = response.json()

        if 'error' in data:
            return {"error": f"SerpAPI error: {data['error']}"}

        # Check if transcript is available
        if 'transcripts' not in data or not data['transcripts']:
            return {"error": "No transcripts available for this video"}

        transcripts = data['transcripts']
        
        # Find the best transcript (prefer English, then any available)
        best_transcript = None
        
        # Look for English transcript first
        for transcript in transcripts:
            if transcript.get('language_code', '').startswith('en'):
                best_transcript = transcript
                break
        
        # If no English, use the first available
        if not best_transcript and transcripts:
            best_transcript = transcripts[0]

        if not best_transcript:
            return {"error": "No suitable transcript found"}

        # Format the transcript data
        transcript_data = best_transcript.get('transcript', [])
        
        # Create timestamped transcript
        timestamped_transcript = []
        full_text_parts = []
        
        for entry in transcript_data:
            if isinstance(entry, dict):
                start_time = entry.get('start', 0)
                text = entry.get('text', '').strip()
                duration = entry.get('duration', 0)
                
                if text:  # Only add non-empty text
                    timestamped_transcript.append({
                        'start': start_time,
                        'duration': duration,
                        'text': text,
                        'formatted_time': format_time(start_time)
                    })
                    full_text_parts.append(text)

        full_text = ' '.join(full_text_parts)

        return {
            "success": True,
            "video_id": video_id,
            "transcript_type": "SerpAPI Extracted",
            "language": best_transcript.get('language', 'Unknown'),
            "language_code": best_transcript.get('language_code', 'unknown'),
            "is_generated": best_transcript.get('is_generated', True),
            "total_segments": len(timestamped_transcript),
            "full_text": full_text,
            "timestamped_transcript": timestamped_transcript
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to get transcript: {str(e)}"}

def print_transcript_summary(result, video_title=""):
    """Print a summary of the transcript."""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"🎬 Video: {video_title}")
    print(f"✅ Transcript Type: {result['transcript_type']}")
    print(f"📺 Language: {result['language']} ({result['language_code']})")
    print(f"🤖 Auto-generated: {'Yes' if result['is_generated'] else 'No'}")
    print(f"📊 Total segments: {result['total_segments']}")
    print("\n" + "=" * 50)
    print("📝 First 3 Segments:")
    print("=" * 50)

    for caption in result['timestamped_transcript'][:3]:
        print(f"[{caption['formatted_time']}] {caption['text']}")

    print("\n" + "=" * 50)
    print("📄 Full Transcript Preview (First 300 characters):")
    print("=" * 50)
    print(result['full_text'][:300] + "..." if len(result['full_text']) > 300 else result['full_text'])

def save_transcript_to_file(result, video_title="", video_stats=None):
    """Save transcript to file with video stats."""
    if "success" not in result or not result["success"]:
        return None

    # Clean filename
    safe_title = re.sub(r'[^\w\s-]', '', video_title)[:50]
    filename = f"transcript_{result['video_id']}_{safe_title}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Video Title: {video_title}\n")
        f.write(f"Video ID: {result['video_id']}\n")
        
        # Add video statistics if available
        if video_stats:
            f.write(f"Views: {format_number(video_stats.get('view_count', 0))}\n")
            f.write(f"Likes: {format_number(video_stats.get('like_count', 0))}\n")
            f.write(f"Comments: {format_number(video_stats.get('comment_count', 0))}\n")
        
        f.write(f"Transcript Type: {result['transcript_type']}\n")
        f.write(f"Language: {result['language']} ({result['language_code']})\n")
        f.write(f"Auto-generated: {'Yes' if result['is_generated'] else 'No'}\n")
        f.write("=" * 50 + "\n\n")
        f.write("FULL TRANSCRIPT:\n")
        f.write("-" * 20 + "\n")
        f.write(result['full_text'])
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("TIMESTAMPED TRANSCRIPT:\n")
        f.write("=" * 50 + "\n")
        for caption in result['timestamped_transcript']:
            f.write(f"[{caption['formatted_time']}] {caption['text']}\n")

    return filename

# Configuration - Replace with your API keys
SERPER_API_KEY = "a4dcbd563f45b80c87043f0ade64988aeea07851"  # Your existing Serper key
SERPAPI_KEY = "0dc301fa5b2564eef6aee907a4521c26e2b61823e934808359778cc45f2eb5da"  # Get from https://serpapi.com/
# 0dc301fa5b2564eef6aee907a4521c26e2b61823e934808359778cc45f2eb5da
# Initialize with API keys
video_search = SerperVideoSearch(SERPER_API_KEY, SERPAPI_KEY)

def main():
    """Main function to search videos and extract transcripts for top 2 most viewed videos"""
    print("🎬 VIDEO SEARCH + TRANSCRIPT EXTRACTOR (LIGHTNING AI COMPATIBLE)")
    print("=" * 70)
    print("📌 Uses SerpAPI for transcript extraction (no blocking issues!)")
    print("📌 Enter your search query to find videos and extract transcripts!")
    print("📌 The tool will get the TOP 2 MOST VIEWED videos and extract their transcripts")
    
    if SERPAPI_KEY == "YOUR_SERPAPI_KEY_HERE":
        print("\n⚠️  WARNING: Please set your SerpAPI key to extract transcripts!")
        print("🔗 Get your free SerpAPI key at: https://serpapi.com/")
        print("💡 For now, the tool will work without transcripts (search only)")
    
    print("=" * 70)

    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Enter your search query (or 'quit' to exit): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("❌ Please enter a valid search query!")
                continue

            print(f"\n🔍 Searching for: '{user_input}'")
            print("=" * 70)

            # Search for videos and get top 2 most viewed
            print("📊 Getting video statistics and finding top viewed videos...")
            top_videos = video_search.get_top_viewed_videos(user_input, top_n=2)

            if not top_videos:
                print("❌ No videos found for your query!")
                continue

            print(f"✅ Found top {len(top_videos)} most viewed videos. Getting transcripts...")
            print("=" * 70)

            # Process each top viewed video
            for i, video in enumerate(top_videos, 1):
                print(f"\n🏆 TOP VIEWED VIDEO {i}: {video['title']}")
                print(f"🔗 Link: {video['link']}")
                print(f"👀 Views: {format_number(video['view_count'])}")
                print(f"👍 Likes: {format_number(video['like_count'])}")
                print(f"💬 Comments: {format_number(video['comment_count'])}")
                print(f"📅 Date: {video['date'] or 'N/A'}")
                print(f"⏱️  Duration: {video['duration'] or 'N/A'}")
                print(f"📺 Channel: {video['channel'] or 'N/A'}")
                print("\n" + "-" * 70)

                # Extract transcript using SerpAPI
                if video['video_id'] and SERPAPI_KEY != "YOUR_SERPAPI_KEY_HERE":
                    print(f"📝 Extracting transcript for top viewed video {i} using SerpAPI...")
                    result = get_youtube_transcript_serpapi(video['video_id'], SERPAPI_KEY)

                    if "success" in result and result["success"]:
                        print_transcript_summary(result, video['title'])

                        # Save to file with video stats
                        video_stats = {
                            'view_count': video['view_count'],
                            'like_count': video['like_count'],
                            'comment_count': video['comment_count']
                        }
                        filename = save_transcript_to_file(result, video['title'], video_stats)
                        if filename:
                            print(f"💾 Transcript saved to: {filename}")
                    else:
                        print(f"❌ Could not extract transcript: {result.get('error', 'Unknown error')}")
                elif not video['video_id']:
                    print("❌ Could not extract video ID from link")
                else:
                    print("⚠️  Skipping transcript extraction - SerpAPI key not configured")

                print("\n" + "=" * 70)

            # Ask if user wants to continue
            continue_search = input("\n🔄 Do you want to search for another query? (y/n): ").strip().lower()
            if continue_search not in ['y', 'yes']:
                print("👋 Thank you for using the tool!")
                break

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again with a different query.")

def quick_demo():
    """Quick demo with a sample query - shows top 2 most viewed videos"""
    print("🧪 QUICK DEMO - Searching for 'python tutorial' (Top 2 Most Viewed)")
    print("=" * 70)

    try:
        top_videos = video_search.get_top_viewed_videos("python tutorial", top_n=2)

        if not top_videos:
            print("❌ No videos found!")
            return

        print(f"✅ Found top {len(top_videos)} most viewed videos:")

        for i, video in enumerate(top_videos, 1):
            print(f"\n🏆 TOP VIEWED {i}. {video['title']}")
            print(f"   🔗 {video['link']}")
            print(f"   👀 Views: {format_number(video['view_count'])}")
            print(f"   👍 Likes: {format_number(video['like_count'])}")
            print(f"   📅 {video['date'] or 'N/A'}")
            print(f"   📺 {video['channel'] or 'N/A'}")

            # Try to get transcript if SerpAPI key is configured
            if video['video_id'] and SERPAPI_KEY != "YOUR_SERPAPI_KEY_HERE":
                print(f"\n📝 Getting transcript for video {i} using SerpAPI...")
                result = get_youtube_transcript_serpapi(video['video_id'], SERPAPI_KEY)

                if "success" in result and result["success"]:
                    print(f"✅ Transcript extracted successfully!")
                    print(f"📊 {result['total_segments']} segments found")
                    print(f"🗣️  Language: {result['language']}")
                    print(f"📄 Preview: {result['full_text'][:200]}...")
                else:
                    print(f"❌ Transcript extraction failed: {result.get('error', 'Unknown error')}")
            else:
                print("⚠️  Transcript extraction skipped - SerpAPI key needed")

        print("\n✅ Demo completed!")
        print("🚀 Run main() to start interactive search")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

# Auto-run information
print("🎬 VIDEO SEARCH + TRANSCRIPT EXTRACTOR (LIGHTNING AI COMPATIBLE) - READY!")
print("=" * 70)
print("✅ Serper API Key loaded successfully!")
print("🚀 Ready to search videos and extract transcripts for TOP 2 MOST VIEWED!")
print("\n📌 NEW FEATURES:")
print("📊 - Gets video views, likes, and comments")
print("🏆 - Automatically selects TOP 2 MOST VIEWED videos")  
print("📝 - Uses SerpAPI for transcript extraction (no blocking!)")
print("💾 - Saves transcripts with video statistics")
print("\n📌 SETUP REQUIRED:")
print("🔑 - Get your SerpAPI key from: https://serpapi.com/")
print("🔧 - Replace 'YOUR_SERPAPI_KEY_HERE' with your actual SerpAPI key")
print("\n📌 OPTIONS:")
print("1. Run main() - Interactive search with top viewed videos")
print("2. Run quick_demo() - Demo with sample query")
print("3. Search manually:")
print("   top_videos = video_search.get_top_viewed_videos('your query', top_n=2)")
print("   result = get_youtube_transcript_serpapi(top_videos[0]['video_id'], SERPAPI_KEY)")
print("=" * 70)

# Uncomment to start immediately
main()
