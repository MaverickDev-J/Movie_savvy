# import requests
# import json
# import re
# from urllib.parse import urlparse, parse_qs

# class YouTubeTranscriptScraper:
#     def __init__(self, api_key, scraper_service="scraperapi"):
#         """
#         Initialize the scraper with API credentials
        
#         Args:
#             api_key (str): Your scraper API key
#             scraper_service (str): Service type - "scraperapi", "apify", or "custom"
#         """
#         self.api_key = api_key
#         self.scraper_service = scraper_service.lower()
        
#         # API endpoints for different services
#         self.endpoints = {
#             "scraperapi": "http://api.scraperapi.com/",
#             "apify": "https://api.apify.com/v2/acts/",
#             "custom": "https://your-custom-api.com/scrape"
#         }
    
#     def extract_video_id(self, youtube_url):
#         """Extract video ID from YouTube URL"""
#         patterns = [
#             r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
#             r'youtube\.com/watch\?.*v=([^&\n?#]+)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, youtube_url)
#             if match:
#                 return match.group(1)
        
#         raise ValueError("Invalid YouTube URL")
    
#     def get_transcript_scraperapi(self, video_id):
#         """Get transcript using ScraperAPI"""
#         transcript_url = f"https://www.youtube.com/api/timedtext?lang=en&v={video_id}"
        
#         params = {
#             'api_key': self.api_key,
#             'url': transcript_url,
#             'render': False
#         }
        
#         response = requests.get(self.endpoints["scraperapi"], params=params)
#         return response
    
#     def get_transcript_apify(self, video_id):
#         """Get transcript using Apify"""
#         # This is a generic structure - you'd need to adapt for specific Apify actors
#         url = f"{self.endpoints['apify']}your-actor-id/run-sync-get-dataset-items"
        
#         headers = {
#             'Authorization': f'Bearer {self.api_key}',
#             'Content-Type': 'application/json'
#         }
        
#         data = {
#             "videoId": video_id,
#             "includeTranscript": True
#         }
        
#         response = requests.post(url, headers=headers, json=data)
#         return response
    
#     def get_transcript_custom(self, youtube_url):
#         """Get transcript using custom API"""
#         params = {
#             'api_key': self.api_key,
#             'url': youtube_url,
#             'extract': 'transcript'
#         }
        
#         response = requests.get(self.endpoints["custom"], params=params)
#         return response
    
#     def parse_transcript_xml(self, xml_content):
#         """Parse XML transcript format"""
#         import xml.etree.ElementTree as ET
        
#         try:
#             root = ET.fromstring(xml_content)
#             transcript = []
            
#             for text_elem in root.findall('.//text'):
#                 start_time = float(text_elem.get('start', 0))
#                 duration = float(text_elem.get('dur', 0))
#                 text = text_elem.text or ""
                
#                 transcript.append({
#                     'start': start_time,
#                     'duration': duration,
#                     'text': text.strip()
#                 })
            
#             return transcript
#         except ET.ParseError as e:
#             print(f"Error parsing XML: {e}")
#             return None
    
#     def parse_transcript_json(self, json_content):
#         """Parse JSON transcript format"""
#         try:
#             data = json.loads(json_content)
            
#             # Adapt this based on your API's response format
#             if 'transcript' in data:
#                 return data['transcript']
#             elif 'captions' in data:
#                 return data['captions']
#             else:
#                 return data
#         except json.JSONDecodeError as e:
#             print(f"Error parsing JSON: {e}")
#             return None
    
#     def scrape_transcript(self, youtube_url):
#         """
#         Main method to scrape transcript from YouTube URL
        
#         Args:
#             youtube_url (str): YouTube video URL
            
#         Returns:
#             list: Transcript data with timestamps
#         """
#         try:
#             video_id = self.extract_video_id(youtube_url)
#             print(f"Extracted video ID: {video_id}")
            
#             # Choose scraping method based on service
#             if self.scraper_service == "scraperapi":
#                 response = self.get_transcript_scraperapi(video_id)
#             elif self.scraper_service == "apify":
#                 response = self.get_transcript_apify(video_id)
#             else:
#                 response = self.get_transcript_custom(youtube_url)
            
#             if response.status_code == 200:
#                 content = response.text
                
#                 # Try to parse as XML first (YouTube's default format)
#                 if content.strip().startswith('<?xml') or content.strip().startswith('<transcript'):
#                     transcript = self.parse_transcript_xml(content)
#                 else:
#                     # Try parsing as JSON
#                     transcript = self.parse_transcript_json(content)
                
#                 return transcript
#             else:
#                 print(f"API request failed with status code: {response.status_code}")
#                 print(f"Response: {response.text}")
#                 return None
                
#         except Exception as e:
#             print(f"Error scraping transcript: {e}")
#             return None
    
#     def format_transcript(self, transcript_data, include_timestamps=True):
#         """Format transcript data into readable text"""
#         if not transcript_data:
#             return "No transcript available"
        
#         formatted_text = []
        
#         for entry in transcript_data:
#             if isinstance(entry, dict):
#                 text = entry.get('text', '')
#                 if include_timestamps and 'start' in entry:
#                     minutes = int(entry['start'] // 60)
#                     seconds = int(entry['start'] % 60)
#                     formatted_text.append(f"[{minutes:02d}:{seconds:02d}] {text}")
#                 else:
#                     formatted_text.append(text)
#             else:
#                 formatted_text.append(str(entry))
        
#         return '\n'.join(formatted_text)


# # Example usage
# def main():
#     # Initialize scraper with your API key
#     API_KEY = "44b94dfeba85e00690db16e898cfd858"
#     scraper = YouTubeTranscriptScraper(API_KEY, "scraperapi")
    
#     # YouTube video URL
#     youtube_url = "https://www.youtube.com/watch?v=okHkUIW46ks"
    
#     print("Scraping transcript...")
#     transcript = scraper.scrape_transcript(youtube_url)
    
#     if transcript:
#         print("\n=== TRANSCRIPT FOUND ===")
#         formatted_transcript = scraper.format_transcript(transcript)
#         print(formatted_transcript)
        
#         # Save to file
#         with open('transcript.txt', 'w', encoding='utf-8') as f:
#             f.write(formatted_transcript)
#         print("\nTranscript saved to 'transcript.txt'")
#     else:
#         print("Failed to retrieve transcript")


# if __name__ == "__main__":
#     main()
import requests
import re
import json
from urllib.parse import urlparse, parse_qs

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
    
    def format_transcript(self, transcript_data):
        """
        Format transcript data into readable text
        
        Args:
            transcript_data (dict): Raw transcript data from API
            
        Returns:
            str: Formatted transcript text
        """
        if not transcript_data.get('success'):
            return f"Error: {transcript_data.get('error', 'Unknown error')}"
        
        data = transcript_data.get('data', {})
        
        # Check if transcripts exists (based on SearchAPI.io response structure)
        if 'transcripts' not in data:
            # Check for error message
            if 'error' in data:
                return f"API Error: {data['error']}"
            return "No transcript available for this video"
        
        transcript_entries = data['transcripts']
        
        if not transcript_entries:
            return "Transcript is empty"
        
        # Format transcript based on SearchAPI.io structure
        formatted_text = []
        for entry in transcript_entries:
            if isinstance(entry, dict):
                text = entry.get('text', '')
                start_time = entry.get('start', 0)
                
                # Convert seconds to MM:SS format
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                
                formatted_text.append(f"{timestamp} {text}")
            elif isinstance(entry, str):
                formatted_text.append(entry)
        
        return '\n'.join(formatted_text)

def main():
    """
    Example usage of the YouTube Transcript Fetcher
    """
    # Replace with your actual SearchAPI.io API key
    API_KEY = "SG2t3x5HGEjqzn217xnHFo76"
    
    # Initialize the fetcher
    fetcher = YouTubeTranscriptFetcher(API_KEY)
    
    # Example YouTube URL
    youtube_url = input("Enter YouTube URL: ").strip()
    
    if not youtube_url:
        print("No URL provided")
        return
    
    print(f"Fetching transcript for: {youtube_url}")
    
    # Get transcript
    result = fetcher.get_transcript(youtube_url)
    
    if result['success']:
        print(f"\nVideo ID: {result['video_id']}")
        print("\n" + "="*50)
        print("TRANSCRIPT:")
        print("="*50)
        
        # Format and display transcript
        formatted_transcript = fetcher.format_transcript(result)
        print(formatted_transcript)
        
        # Optionally save to file
        save_to_file = input("\nSave transcript to file? (y/n): ").lower().strip()
        if save_to_file == 'y':
            filename = f"transcript_{result['video_id']}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_transcript)
            print(f"Transcript saved to {filename}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()