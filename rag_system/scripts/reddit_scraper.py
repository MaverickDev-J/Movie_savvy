
# import cloudscraper
# from bs4 import BeautifulSoup
# import requests
# import praw
# import argparse
# import re
# import time
# import json
# import os
# from datetime import datetime
# import uuid
# import logging
# from pathlib import Path
# from dotenv import load_dotenv
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import warnings
# warnings.filterwarnings("ignore")

# load_dotenv()

# # Configuration
# REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
# REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
# REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# # Relevant subreddits for content filtering
# RELEVANT_SUBREDDITS = {
#     'anime': ['anime', 'animesuggest', 'animediscussion', 'manga', 'onepiece', 'naruto', 'attackontitan'],
#     'kdrama': ['kdrama', 'koreandramas', 'kdramas', 'asiandrama'],
#     'kmovie': ['koreanmovies', 'asianmovies', 'worldcinema'],
#     'bollywood': ['bollywood', 'indianmovies', 'bollywoodmemes', 'hindicinema'],
#     'hollywood': ['movies', 'moviesuggestions', 'filmmakers', 'moviedetails', 'marvelstudios', 'dc_cinematic'],
#     'tv_series': ['television', 'tvshows', 'netflix', 'amazonprime', 'hbo'],
#     'general': ['movies', 'television', 'entertainment', 'popculture']
# }

# # Plot-related keywords for quality filtering
# PLOT_KEYWORDS = {
#     'story_elements': ['plot', 'story', 'storyline', 'narrative', 'character development', 'protagonist', 'antagonist'],
#     'ending_keywords': ['ending', 'finale', 'conclusion', 'final episode', 'last episode', 'climax', 'resolution'],
#     'plot_devices': ['twist', 'reveal', 'spoiler', 'plot twist', 'cliffhanger', 'foreshadowing', 'subplot'],
#     'discussion_terms': ['what happens', 'explain', 'explanation', 'theory', 'analysis', 'interpretation', 'meaning'],
#     'emotional_terms': ['shocking', 'surprising', 'heartbreaking', 'emotional', 'tragic', 'dramatic', 'intense'],
#     'season_terms': ['season finale', 'series finale', 'final season', 'last season', 'season ending']
# }

# # Informational/promotional content indicators (to filter out)
# PROMOTIONAL_KEYWORDS = [
#     'cast', 'casting', 'actor', 'actress', 'director', 'producer', 'filming', 'production',
#     'release date', 'premiere', 'streaming', 'watch online', 'where to watch', 'episode count',
#     'season count', 'rating', 'imdb', 'metacritic', 'review score', 'budget', 'box office',
#     'trailer', 'teaser', 'official', 'announcement', 'confirmed', 'renewed', 'cancelled'
# ]

# def extract_drama_name_and_type(query):
#     """Extract drama/movie name and type from query"""
#     query_lower = query.lower().strip()
    
#     # Define content types
#     content_types = {
#         'anime': ['anime'],
#         'manga': ['manga'],
#         'kdrama': ['kdrama', 'korean drama', 'k-drama'],
#         'kmovie': ['kmovie', 'korean movie', 'k-movie'],
#         'bollywood': ['bollywood', 'hindi movie', 'hindi film'],
#         'hollywood': ['hollywood', 'american movie', 'american film'],
#         'netflix': ['netflix series', 'netflix show'],
#         'tv_series': ['tv series', 'tv show', 'series', 'show'],
#         'movie': ['movie', 'film']
#     }
    
#     detected_type = 'unknown'
    
#     # Check for explicit type mentions
#     for content_type, keywords in content_types.items():
#         for keyword in keywords:
#             if keyword in query_lower:
#                 detected_type = content_type
#                 # Remove the type keyword from query for name extraction
#                 query_lower = query_lower.replace(keyword, '').strip()
#                 break
#         if detected_type != 'unknown':
#             break
    
#     # Enhanced patterns for name extraction
#     patterns = [
#         # Direct title patterns
#         r'(?:plot|story|ending|explanation|summary|spoilers?)\s+(?:of\s+)?(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?(?:\s+(?:movie|film|drama|series|show))?$',
#         r'(?:explain|tell|describe)\s+(?:me\s+)?(?:the\s+)?(?:plot|story|ending)\s+(?:of\s+)?(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?(?:\s+(?:movie|film|drama|series|show))?$',
#         r'what\s+(?:is|happens|happened)\s+(?:in\s+)?(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?(?:\s+(?:movie|film|drama|series|show))?$',
#         # Title followed by plot-related words
#         r'^([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?\s+(?:plot|story|ending|explanation|summary|spoilers?)$',
#         # Generic title extraction
#         r'^(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?$'
#     ]
    
#     drama_name = None
    
#     # Try patterns
#     for pattern in patterns:
#         if match := re.search(pattern, query_lower, re.IGNORECASE):
#             extracted = match.group(1).strip()
            
#             # Clean the extracted name
#             extracted = re.sub(r'^(the\s+)', '', extracted, flags=re.IGNORECASE)
#             extracted = re.sub(r'\s+(movie|film|drama|series|show)$', '', extracted, flags=re.IGNORECASE)
#             extracted = re.sub(r'[^\w\s\-\'\":.]', '', extracted).strip()
            
#             # Filter out generic words
#             stop_words = {'plot', 'story', 'ending', 'explanation', 'summary', 'spoiler', 'spoilers', 
#                          'explain', 'tell', 'describe', 'what', 'is', 'happens', 'happened', 'in', 'of', 'the', 'me'}
            
#             words = extracted.split()
#             meaningful_words = [w for w in words if w.lower() not in stop_words and len(w) > 1]
            
#             if meaningful_words and len(' '.join(meaningful_words)) > 2:
#                 drama_name = ' '.join(meaningful_words).title()
#                 break
    
#     # Fallback: extract meaningful words from original query
#     if not drama_name:
#         original_words = re.findall(r'\b[a-zA-Z0-9]+\b', query)
#         stop_words = {'plot', 'story', 'ending', 'explanation', 'summary', 'spoiler', 'spoilers', 
#                      'explain', 'tell', 'describe', 'what', 'is', 'happens', 'happened', 'in', 'of', 
#                      'the', 'me', 'movie', 'film', 'drama', 'series', 'show', 'anime', 'manga'}
        
#         meaningful_words = [w for w in original_words if w.lower() not in stop_words and len(w) > 1]
#         drama_name = ' '.join(meaningful_words[:4]).title() if meaningful_words else "Unknown Title"
    
#     return drama_name, detected_type

# def calculate_plot_relevance_score(text):
#     """Calculate how relevant the text is to plot/story discussions"""
#     text_lower = text.lower()
    
#     # Count plot-related keywords
#     plot_score = 0
#     total_keywords = 0
    
#     for category, keywords in PLOT_KEYWORDS.items():
#         category_score = 0
#         for keyword in keywords:
#             if keyword in text_lower:
#                 category_score += 1
        
#         # Weight different categories
#         if category == 'story_elements':
#             plot_score += category_score * 2
#         elif category == 'ending_keywords':
#             plot_score += category_score * 3
#         elif category == 'plot_devices':
#             plot_score += category_score * 2
#         else:
#             plot_score += category_score
        
#         total_keywords += len(keywords)
    
#     # Penalize promotional content
#     promotional_penalty = 0
#     for keyword in PROMOTIONAL_KEYWORDS:
#         if keyword in text_lower:
#             promotional_penalty += 1
    
#     # Calculate final score (0-100)
#     base_score = min((plot_score / max(total_keywords * 0.1, 1)) * 100, 100)
#     final_score = max(0, base_score - (promotional_penalty * 10))
    
#     return final_score

# def is_relevant_subreddit(url, content_type):
#     """Check if the post is from a relevant subreddit"""
#     try:
#         # Extract subreddit from URL
#         subreddit_match = re.search(r'reddit\.com/r/([^/]+)', url)
#         if not subreddit_match:
#             return False
        
#         subreddit = subreddit_match.group(1).lower()
        
#         # Check against relevant subreddits
#         relevant_subs = RELEVANT_SUBREDDITS.get(content_type, []) + RELEVANT_SUBREDDITS['general']
#         return any(sub in subreddit for sub in relevant_subs)
#     except:
#         return False

# def is_quality_content(text, min_words=20, max_words=500, min_plot_score=20):
#     """Enhanced quality check with plot relevance"""
#     if not text or len(text.strip()) < 10:
#         return False
    
#     words = text.split()
#     word_count = len(words)
    
#     # Length check
#     if word_count < min_words or word_count > max_words:
#         return False
    
#     # Plot relevance check
#     plot_score = calculate_plot_relevance_score(text)
#     if plot_score < min_plot_score:
#         return False
    
#     # Remove obvious meme/joke comments
#     meme_patterns = [
#         r'^(lol|lmao|haha|omg|wow|this|same|exactly|agreed?|totally|definitely|absolutely)\.?$',
#         r'^(upvote|downvote|thanks|thank you|nice|good|great|awesome|bad|terrible)\.?$',
#         r'^(\+1|this\s+is\s+it|this\s+right\s+here|came\s+here\s+to\s+say\s+this)\.?$'
#     ]
    
#     text_lower = text.lower().strip()
#     if any(re.match(pattern, text_lower) for pattern in meme_patterns):
#         return False
    
#     # Check for substantial content (not just reactions)
#     content_indicators = ['because', 'however', 'although', 'when', 'where', 'why', 'how', 'what', 'which', 'who']
#     has_substance = any(indicator in text_lower for indicator in content_indicators)
    
#     return has_substance or word_count > 30

# def is_bot_account(author_name):
#     """Check if account is likely a bot"""
#     if not author_name or author_name in ['[deleted]', '[removed]', 'AutoModerator']:
#         return True
    
#     bot_indicators = ['bot', 'auto', 'moderator', 'reminder', 'converter']
#     return any(indicator in author_name.lower() for indicator in bot_indicators)

# def get_reddit_content_enhanced(url, scraper, content_type):
#     """Get Reddit content with enhanced filtering"""
#     try:
#         # Check subreddit relevance first
#         if not is_relevant_subreddit(url, content_type):
#             return None
        
#         # Try Reddit API first
#         reddit = praw.Reddit(
#             client_id=REDDIT_CLIENT_ID, 
#             client_secret=REDDIT_CLIENT_SECRET, 
#             user_agent=REDDIT_USER_AGENT, 
#             read_only=True
#         )
        
#         if '/comments/' in url:
#             post_id = url.split('/comments/')[1].split('/')[0]
#             submission = reddit.submission(id=post_id)
            
#             # Check post quality
#             if submission.score < 5 or is_bot_account(submission.author.name if submission.author else None):
#                 return None
            
#             post_text = f"{submission.title} {submission.selftext}".strip()
#             if not is_quality_content(post_text, min_words=10, max_words=1000, min_plot_score=15):
#                 return None
            
#             content = {
#                 "title": submission.title,
#                 "post_content": submission.selftext or "No text content",
#                 "post_score": submission.score,
#                 "author": submission.author.name if submission.author else "[deleted]",
#                 "plot_relevance_score": calculate_plot_relevance_score(post_text),
#                 "comments": []
#             }
            
#             # Get and filter comments
#             submission.comments.replace_more(limit=0)
#             all_comments = [c for c in submission.comments.list() if hasattr(c, 'body') and c.body != '[deleted]']
            
#             # Filter comments by quality and plot relevance
#             quality_comments = []
#             for comment in all_comments:
#                 if (comment.score >= 3 and 
#                     not is_bot_account(comment.author.name if comment.author else None) and
#                     is_quality_content(comment.body, min_plot_score=25)):
#                     quality_comments.append(comment)
            
#             # Sort by combination of score and plot relevance
#             def comment_score(c):
#                 plot_score = calculate_plot_relevance_score(c.body)
#                 return (c.score * 0.3) + (plot_score * 0.7)
            
#             top_comments = sorted(quality_comments, key=comment_score, reverse=True)[:5]
            
#             for idx, comment in enumerate(top_comments, 1):
#                 content["comments"].append({
#                     "comment_number": idx,
#                     "body": comment.body,
#                     "score": comment.score,
#                     "author": comment.author.name if comment.author else "[deleted]",
#                     "plot_relevance_score": calculate_plot_relevance_score(comment.body)
#                 })
            
#             return content
            
#     except Exception as e:
#         print(f"⚠ Reddit API failed for {url}: {e}")
#         return None

# def clean_and_filter_text_enhanced(text):
#     """Enhanced text cleaning and filtering"""
#     if not isinstance(text, str) or len(text.strip()) < 10:
#         return None
    
#     # Clean text
#     text = re.sub(r'#\s*SPOILER ALERT\s*|\n+|\*\w+\*', ' ', text, flags=re.IGNORECASE)
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Additional quality check with plot relevance
#     if not is_quality_content(text, min_plot_score=20):
#         return None
    
#     return text

# class SmartRedditScraperBotWithEmbeddings:
#     def __init__(self, delay=2, output_dir=str(Path(__file__).resolve().parent.parent / "data" / "processed" / "reddit_data")):
#         self.scraper = cloudscraper.create_scraper()
#         self.delay = delay
#         self.output_dir = output_dir
#         self.similarity_threshold = 0.6
#         self.plot_relevance_threshold = 30  # Minimum plot relevance score
        
#         # Initialize embedding model
#         print("🤖 Loading embedding model...")
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#         print("✅ Model loaded successfully!")
        
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#     def compute_embeddings_batch(self, texts):
#         """Compute embeddings in batches for efficiency"""
#         if not texts:
#             return np.array([])
        
#         print(f"🔢 Computing embeddings for {len(texts)} texts...")
#         embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
#         return embeddings

#     def calculate_similarity(self, query_embedding, content_embeddings):
#         """Calculate cosine similarity between query and content"""
#         if len(content_embeddings) == 0:
#             return []
        
#         similarities = cosine_similarity([query_embedding], content_embeddings)[0]
#         return similarities

#     def create_fallback_searches(self, query, drama_name, content_type):
#         """Create fallback search queries for better results"""
#         fallback_queries = []
        
#         # More specific plot-focused searches
#         plot_terms = ["ending explained", "finale explanation", "plot summary", "spoiler discussion", "ending meaning"]
        
#         for term in plot_terms:
#             fallback_queries.append(f"{drama_name} {term} reddit")
        
#         # Season-specific searches if applicable
#         if "season" in query.lower():
#             season_match = re.search(r'season\s*(\d+)', query.lower())
#             if season_match:
#                 season_num = season_match.group(1)
#                 fallback_queries.append(f"{drama_name} season {season_num} ending explained reddit")
#                 fallback_queries.append(f"{drama_name} s{season_num} finale discussion reddit")
        
#         return fallback_queries[:3]  # Limit to 3 fallback searches

#     def search_and_preprocess_with_embeddings(self, query, num_posts=3):
#         """Search Reddit, preprocess, and calculate embeddings with enhanced quality filtering"""
#         drama_name, content_type = extract_drama_name_and_type(query)
#         print(f"🎭 Detected: '{drama_name}' ({content_type}) | 🔍 Searching...")
        
#         if not SERPER_API_KEY:
#             print("⚠ No SERPER API key set")
#             return {}

#         try:
#             processed_data = []
            
#             # Primary search
#             processed_data.extend(self._search_with_query(query, drama_name, content_type, num_posts))
            
#             # If we don't have enough quality results, try fallback searches
#             if len(processed_data) < num_posts:
#                 print(f"🔄 Primary search yielded {len(processed_data)} results. Trying fallback searches...")
                
#                 fallback_queries = self.create_fallback_searches(query, drama_name, content_type)
#                 for fallback_query in fallback_queries:
#                     if len(processed_data) >= num_posts:
#                         break
                    
#                     print(f"🔍 Fallback search: '{fallback_query}'")
#                     fallback_data = self._search_with_query(fallback_query, drama_name, content_type, num_posts - len(processed_data))
#                     processed_data.extend(fallback_data)

#             if not processed_data:
#                 print("❌ No quality content found")
#                 return {}

#             # Compute embeddings and similarity
#             print(f"\n🧠 Processing {len(processed_data)} pieces of content...")
            
#             # Extract texts for embedding
#             texts = [item['text'] for item in processed_data]
            
#             # Compute query embedding
#             query_embedding = self.embedding_model.encode([query])[0]
            
#             # Compute content embeddings in batch
#             content_embeddings = self.compute_embeddings_batch(texts)
            
#             # Calculate similarities
#             similarities = self.calculate_similarity(query_embedding, content_embeddings)
            
#             # Add similarity scores to processed data
#             for i, item in enumerate(processed_data):
#                 item['similarity_score'] = float(similarities[i])
            
#             # Enhanced filtering: combine similarity and plot relevance
#             filtered_data = []
#             for item in processed_data:
#                 similarity_ok = item['similarity_score'] > self.similarity_threshold
#                 plot_relevance_ok = item.get('plot_relevance_score', 0) > self.plot_relevance_threshold
                
#                 # Combined score: 60% similarity + 40% plot relevance
#                 combined_score = (item['similarity_score'] * 0.6) + (item.get('plot_relevance_score', 0) / 100 * 0.4)
#                 item['combined_score'] = combined_score
                
#                 if similarity_ok and plot_relevance_ok:
#                     filtered_data.append(item)
            
#             # Sort by combined score
#             filtered_data.sort(key=lambda x: x['combined_score'], reverse=True)
            
#             # Create final output
#             output = {
#                 "query": query,
#                 "drama_name": drama_name,
#                 "content_type": content_type,
#                 "source": "Reddit",
#                 "similarity_threshold": self.similarity_threshold,
#                 "plot_relevance_threshold": self.plot_relevance_threshold,
#                 "total_content_found": len(processed_data),
#                 "relevant_content_count": len(filtered_data),
#                 "data": filtered_data
#             }
            
#             # Save and display
#             self.save_data(drama_name, output)
#             self.display_enhanced_results(output)
            
#             return output

#         except Exception as e:
#             print(f"❌ Error: {e}")
#             return {}

#     def _search_with_query(self, search_query, drama_name, content_type, max_posts):
#         """Internal method to search with a specific query"""
#         try:
#             # Create search query with subreddit targeting
#             relevant_subs = RELEVANT_SUBREDDITS.get(content_type, []) + RELEVANT_SUBREDDITS['general']
#             subreddit_query = ' OR '.join([f'site:reddit.com/r/{sub}' for sub in relevant_subs[:5]])
#             full_search_query = f"{search_query} ({subreddit_query})"
            
#             # Search Reddit posts
#             response = requests.post(
#                 "https://google.serper.dev/search", 
#                 headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
#                 data=json.dumps({"q": full_search_query, "num": 15})
#             )
            
#             search_results = response.json()
#             processed_data = []
#             posts_processed = 0
            
#             if 'organic' in search_results:
#                 for result in search_results['organic']:
#                     if posts_processed >= max_posts:
#                         break
                        
#                     if 'reddit.com' in result.get('link', ''):
#                         reddit_content = get_reddit_content_enhanced(result['link'], self.scraper, content_type)
                        
#                         if reddit_content:
#                             # Process post
#                             post_text = f"{reddit_content['title']} {reddit_content['post_content']}".strip()
#                             if cleaned_post := clean_and_filter_text_enhanced(post_text):
#                                 processed_data.append({
#                                     "content_id": f"post_{posts_processed + 1}",
#                                     "text": cleaned_post,
#                                     "plot_relevance_score": reddit_content.get('plot_relevance_score', 0),
#                                     "metadata": {
#                                         "type": "post", 
#                                         "score": reddit_content['post_score'],
#                                         "author": reddit_content['author'],
#                                         "timestamp": datetime.now().isoformat()
#                                     }
#                                 })
                            
#                             # Process comments
#                             for comment in reddit_content.get('comments', []):
#                                 if cleaned_comment := clean_and_filter_text_enhanced(comment['body']):
#                                     processed_data.append({
#                                         "content_id": f"post_{posts_processed + 1}_comment_{comment['comment_number']}",
#                                         "text": cleaned_comment,
#                                         "plot_relevance_score": comment.get('plot_relevance_score', 0),
#                                         "metadata": {
#                                             "type": "comment", 
#                                             "score": comment['score'],
#                                             "author": comment['author'],
#                                             "timestamp": datetime.now().isoformat()
#                                         }
#                                     })
                            
#                             posts_processed += 1
                        
#                         time.sleep(self.delay)
            
#             return processed_data
            
#         except Exception as e:
#             print(f"⚠ Search error: {e}")
#             return []

#     def save_data(self, drama_name, data):
#         """Save preprocessed data with embeddings"""
#         safe_name = re.sub(r'[^\w\s-]', '', drama_name).replace(' ', '_')
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"reddit_enhanced_{safe_name}_{timestamp}.json"
        
#         with open(os.path.join(self.output_dir, filename), 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
#         print(f"💾 Saved: {filename}")

#     def display_enhanced_results(self, data):
#         """Display results with enhanced scoring"""
#         print(f"\n🎯 '{data['drama_name'].upper()}' ({data['content_type']})")
#         print(f"📊 Found {data['total_content_found']} total | {data['relevant_content_count']} high-quality (Similarity>{data['similarity_threshold']}, Plot Relevance>{data['plot_relevance_threshold']})")
#         print("="*80)
        
#         if not data['data']:
#             print("❌ No content meeting quality thresholds found")
#             return
        
#         for i, item in enumerate(data['data'][:10], 1):  # Show top 10
#             similarity = item['similarity_score']
#             plot_score = item.get('plot_relevance_score', 0)
#             combined_score = item.get('combined_score', 0)
#             content_type = item['metadata']['type'].upper()
#             score = item['metadata']['score']
#             author = item['metadata']['author']
            
#             # Quality indicator
#             if combined_score > 0.8:
#                 quality_indicator = "🟢 EXCELLENT"
#             elif combined_score > 0.7:
#                 quality_indicator = "🟡 GOOD"
#             elif combined_score > 0.6:
#                 quality_indicator = "🟠 FAIR"
#             else:
#                 quality_indicator = "🔴 POOR"
            
#             print(f"\n{quality_indicator} #{i}")
#             print(f"📈 Combined: {combined_score:.3f} | Similarity: {similarity:.3f} | Plot: {plot_score:.1f}%")
#             print(f"📝 {content_type} | Score: {score} | Author: {author}")
#             print(f"💬 {item['text'][:200]}{'...' if len(item['text']) > 200 else ''}")
#             print("-" * 80)

# # Usage Example
# if __name__ == "__main__":
#     bot = SmartRedditScraperBotWithEmbeddings()
#     # Example queries
#     queries = [   
#         "explain the ending of inception"]
#     for query in queries:
#         print(f"\n🚀 Processing: '{query}'")
#         result = bot.search_and_preprocess_with_embeddings(query, num_posts=3)
#         print("="*100)









import cloudscraper
import requests
import praw
import argparse
import re
import time
import json
import os
from datetime import datetime
from pathlib import Path
import logging
import yaml
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configuration paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Reddit credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Logging setup
reddit_log_path = BASE_DIR / config["pipeline"]["logging"]["log_dir"] / config["pipeline"]["logging"]["reddit_log_file"]
logging.basicConfig(
    level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(reddit_log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Relevant subreddits for content filtering
RELEVANT_SUBREDDITS = {
    'anime': ['anime', 'animesuggest', 'animediscussion', 'manga', 'onepiece', 'naruto', 'attackontitan'],
    'kdrama': ['kdrama', 'koreandramas', 'kdramas', 'asiandrama'],
    'kmovie': ['koreanmovies', 'asianmovies', 'worldcinema'],
    'bollywood': ['bollywood', 'indianmovies', 'bollywoodmemes', 'hindicinema'],
    'hollywood': ['movies', 'moviesuggestions', 'filmmakers', 'moviedetails', 'marvelstudios', 'dc_cinematic'],
    'tv_series': ['television', 'tvshows', 'netflix', 'amazonprime', 'hbo'],
    'general': ['movies', 'television', 'entertainment', 'popculture']
}

# Plot-related keywords for quality filtering
PLOT_KEYWORDS = {
    'story_elements': ['plot', 'story', 'storyline', 'narrative', 'character development', 'protagonist', 'antagonist'],
    'ending_keywords': ['ending', 'finale', 'conclusion', 'final episode', 'last episode', 'climax', 'resolution'],
    'plot_devices': ['twist', 'reveal', 'spoiler', 'plot twist', 'cliffhanger', 'foreshadowing', 'subplot'],
    'discussion_terms': ['what happens', 'explain', 'explanation', 'theory', 'analysis', 'interpretation', 'meaning'],
    'emotional_terms': ['shocking', 'surprising', 'heartbreaking', 'emotional', 'tragic', 'dramatic', 'intense'],
    'season_terms': ['season finale', 'series finale', 'final season', 'last season', 'season ending']
}

# Informational/promotional content indicators (to filter out)
PROMOTIONAL_KEYWORDS = [
    'cast', 'casting', 'actor', 'actress', 'director', 'producer', 'filming', 'production',
    'release date', 'premiere', 'streaming', 'watch online', 'where to watch', 'episode count',
    'season count', 'rating', 'imdb', 'metacritic', 'review score', 'budget', 'box office',
    'trailer', 'teaser', 'official', 'announcement', 'confirmed', 'renewed', 'cancelled'
]

def extract_drama_name_and_type(query):
    """Extract drama/movie name and type from query"""
    query_lower = query.lower().strip()
    content_types = {
        'anime': ['anime'],
        'manga': ['manga'],
        'kdrama': ['kdrama', 'korean drama', 'k-drama'],
        'kmovie': ['kmovie', 'korean movie', 'k-movie'],
        'bollywood': ['bollywood', 'hindi movie', 'hindi film'],
        'hollywood': ['hollywood', 'american movie', 'american film'],
        'netflix': ['netflix series', 'netflix show'],
        'tv_series': ['tv series', 'tv show', 'series', 'show'],
        'movie': ['movie', 'film']
    }
    
    detected_type = 'unknown'
    for content_type, keywords in content_types.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected_type = content_type
                query_lower = query_lower.replace(keyword, '').strip()
                break
        if detected_type != 'unknown':
            break
    
    patterns = [
        r'(?:plot|story|ending|explanation|summary|spoilers?)\s+(?:of\s+)?(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?(?:\s+(?:movie|film|drama|series|show))?$',
        r'(?:explain|tell|describe)\s+(?:me\s+)?(?:the\s+)?(?:plot|story|ending)\s+(?:of\s+)?(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?(?:\s+(?:movie|film|drama|series|show))?$',
        r'what\s+(?:is|happens|happened)\s+(?:in\s+)?(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?(?:\s+(?:movie|film|drama|series|show))?$',
        r'^([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?\s+(?:plot|story|ending|explanation|summary|spoilers?)$',
        r'^(?:the\s+)?([a-zA-Z0-9\s\-\'\":]+?)(?:\s+(?:season\s+\d+|vol\s+\d+|part\s+\d+))?$'
    ]
    
    drama_name = None
    for pattern in patterns:
        if match := re.search(pattern, query_lower, re.IGNORECASE):
            extracted = match.group(1).strip()
            extracted = re.sub(r'^(the\s+)', '', extracted, flags=re.IGNORECASE)
            extracted = re.sub(r'\s+(movie|film|drama|series|show)$', '', extracted, flags=re.IGNORECASE)
            extracted = re.sub(r'[^\w\s\-\'\":.]', '', extracted).strip()
            stop_words = {'plot', 'story', 'ending', 'explanation', 'summary', 'spoiler', 'spoilers', 
                         'explain', 'tell', 'describe', 'what', 'is', 'happens', 'happened', 'in', 'of', 'the', 'me'}
            words = extracted.split()
            meaningful_words = [w for w in words if w.lower() not in stop_words and len(w) > 1]
            if meaningful_words and len(' '.join(meaningful_words)) > 2:
                drama_name = ' '.join(meaningful_words).title()
                break
    
    if not drama_name:
        original_words = re.findall(r'\b[a-zA-Z0-9]+\b', query)
        stop_words = {'plot', 'story', 'ending', 'explanation', 'summary', 'spoiler', 'spoilers', 
                     'explain', 'tell', 'describe', 'what', 'is', 'happens', 'happened', 'in', 'of', 
                     'the', 'me', 'movie', 'film', 'drama', 'series', 'show', 'anime', 'manga'}
        meaningful_words = [w for w in original_words if w.lower() not in stop_words and len(w) > 1]
        drama_name = ' '.join(meaningful_words[:4]).title() if meaningful_words else "Unknown Title"
    
    return drama_name, detected_type

def calculate_plot_relevance_score(text):
    """Calculate how relevant the text is to plot/story discussions"""
    text_lower = text.lower()
    plot_score = 0
    total_keywords = 0
    for category, keywords in PLOT_KEYWORDS.items():
        category_score = 0
        for keyword in keywords:
            if keyword in text_lower:
                category_score += 1
        if category == 'story_elements':
            plot_score += category_score * 2
        elif category == 'ending_keywords':
            plot_score += category_score * 3
        elif category == 'plot_devices':
            plot_score += category_score * 2
        else:
            plot_score += category_score
        total_keywords += len(keywords)
    
    promotional_penalty = 0
    for keyword in PROMOTIONAL_KEYWORDS:
        if keyword in text_lower:
            promotional_penalty += 1
    
    base_score = min((plot_score / max(total_keywords * 0.1, 1)) * 100, 100)
    final_score = max(0, base_score - (promotional_penalty * 10))
    return final_score

def is_relevant_subreddit(url, content_type):
    """Check if the post is from a relevant subreddit"""
    try:
        subreddit_match = re.search(r'reddit\.com/r/([^/]+)', url)
        if not subreddit_match:
            return False
        subreddit = subreddit_match.group(1).lower()
        relevant_subs = RELEVANT_SUBREDDITS.get(content_type, []) + RELEVANT_SUBREDDITS['general']
        return any(sub in subreddit for sub in relevant_subs)
    except:
        return False

def is_quality_content(text, min_words=20, max_words=500, min_plot_score=20):
    """Enhanced quality check with plot relevance"""
    if not text or len(text.strip()) < 10:
        return False
    words = text.split()
    word_count = len(words)
    if word_count < min_words or word_count > max_words:
        return False
    plot_score = calculate_plot_relevance_score(text)
    if plot_score < min_plot_score:
        return False
    meme_patterns = [
        r'^(lol|lmao|haha|omg|wow|this|same|exactly|agreed?|totally|definitely|absolutely)\.?$',
        r'^(upvote|downvote|thanks|thank you|nice|good|great|awesome|bad|terrible)\.?$',
        r'^(\+1|this\s+is\s+it|this\s+right\s+here|came\s+here\s+to\s+say\s+this)\.?$'
    ]
    text_lower = text.lower().strip()
    if any(re.match(pattern, text_lower) for pattern in meme_patterns):
        return False
    content_indicators = ['because', 'however', 'although', 'when', 'where', 'why', 'how', 'what', 'which', 'who']
    has_substance = any(indicator in text_lower for indicator in content_indicators)
    return has_substance or word_count > 30

def is_bot_account(author_name):
    """Check if account is likely a bot"""
    if not author_name or author_name in ['[deleted]', '[removed]', 'AutoModerator']:
        return True
    bot_indicators = ['bot', 'auto', 'moderator', 'reminder', 'converter']
    return any(indicator in author_name.lower() for indicator in bot_indicators)

def get_reddit_content_enhanced(url, scraper, content_type):
    """Get Reddit content with enhanced filtering"""
    try:
        if not is_relevant_subreddit(url, content_type):
            return None
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID, 
            client_secret=REDDIT_CLIENT_SECRET, 
            user_agent=REDDIT_USER_AGENT, 
            read_only=True
        )
        if '/comments/' in url:
            post_id = url.split('/comments/')[1].split('/')[0]
            submission = reddit.submission(id=post_id)
            if submission.score < 5 or is_bot_account(submission.author.name if submission.author else None):
                return None
            post_text = f"{submission.title} {submission.selftext}".strip()
            if not is_quality_content(post_text, min_words=10, max_words=1000, min_plot_score=15):
                return None
            content = {
                "title": submission.title,
                "post_content": submission.selftext or "No text content",
                "post_score": submission.score,
                "author": submission.author.name if submission.author else "[deleted]",
                "plot_relevance_score": calculate_plot_relevance_score(post_text),
                "comments": []
            }
            submission.comments.replace_more(limit=0)
            all_comments = [c for c in submission.comments.list() if hasattr(c, 'body') and c.body != '[deleted]']
            quality_comments = []
            for comment in all_comments:
                if (comment.score >= 3 and 
                    not is_bot_account(comment.author.name if comment.author else None) and
                    is_quality_content(comment.body, min_plot_score=25)):
                    quality_comments.append(comment)
            def comment_score(c):
                plot_score = calculate_plot_relevance_score(c.body)
                return (c.score * 0.3) + (plot_score * 0.7)
            top_comments = sorted(quality_comments, key=comment_score, reverse=True)[:5]
            for idx, comment in enumerate(top_comments, 1):
                content["comments"].append({
                    "comment_number": idx,
                    "body": comment.body,
                    "score": comment.score,
                    "author": comment.author.name if comment.author else "[deleted]",
                    "plot_relevance_score": calculate_plot_relevance_score(comment.body)
                })
            return content
    except Exception as e:
        logger.error(f"Reddit API failed for {url}: {e}")
        return None

def clean_and_filter_text_enhanced(text):
    """Enhanced text cleaning and filtering"""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return None
    text = re.sub(r'#\s*SPOILER ALERT\s*|\n+|\*\w+\*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    if not is_quality_content(text, min_plot_score=20):
        return None
    return text

class SmartRedditScraperBotWithEmbeddings:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.delay = 2
        self.output_dir = str(BASE_DIR / config["data"]["reddit_dir"])
        self.similarity_threshold = 0.6
        self.plot_relevance_threshold = 30
        logger.info("Initializing Reddit scraper...")
        self.embedding_model = SentenceTransformer('intfloat/e5-base-v2')
        logger.info("Embedding model loaded successfully")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.embedding_cache = {} if config["reddit"]["scraper"]["embedding_cache"] else None

    def compute_embeddings_batch(self, texts):
        """Compute embeddings in batches for efficiency"""
        if not texts:
            return np.array([])
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        return embeddings

    def calculate_similarity(self, query_embedding, content_embeddings):
        """Calculate cosine similarity between query and content"""
        if len(content_embeddings) == 0:
            return []
        similarities = cosine_similarity([query_embedding], content_embeddings)[0]
        return similarities

    def create_fallback_searches(self, query, drama_name, content_type):
        """Create fallback search queries for better results"""
        fallback_queries = []
        plot_terms = ["ending explained", "finale explanation", "plot summary", "spoiler discussion", "ending meaning"]
        for term in plot_terms:
            fallback_queries.append(f"{drama_name} {term} reddit")
        if "season" in query.lower():
            season_match = re.search(r'season\s*(\d+)', query.lower())
            if season_match:
                season_num = season_match.group(1)
                fallback_queries.append(f"{drama_name} season {season_num} ending explained reddit")
                fallback_queries.append(f"{drama_name} s{season_num} finale discussion reddit")
        return fallback_queries[:3]

    def search_and_preprocess_with_embeddings(self, query, num_posts):
        """Search Reddit, preprocess, and calculate embeddings with enhanced quality filtering"""
        drama_name, content_type = extract_drama_name_and_type(query)
        logger.info(f"Detected: '{drama_name}' ({content_type}) | Searching...")
        if not SERPER_API_KEY:
            logger.error("No SERPER API key set")
            return {}
        try:
            processed_data = []
            processed_data.extend(self._search_with_query(query, drama_name, content_type, num_posts))
            if len(processed_data) < num_posts:
                logger.info(f"Primary search yielded {len(processed_data)} results. Trying fallback searches...")
                fallback_queries = self.create_fallback_searches(query, drama_name, content_type)
                for fallback_query in fallback_queries:
                    if len(processed_data) >= num_posts:
                        break
                    logger.info(f"Fallback search: '{fallback_query}'")
                    fallback_data = self._search_with_query(fallback_query, drama_name, content_type, num_posts - len(processed_data))
                    processed_data.extend(fallback_data)
            if not processed_data:
                logger.warning("No quality content found")
                return {}
            texts = [item['text'] for item in processed_data]
            if self.embedding_cache is not None:
                query_key = query.lower()
                if query_key in self.embedding_cache:
                    logger.info("Using cached query embedding")
                    query_embedding = self.embedding_cache[query_key]
                else:
                    query_embedding = self.embedding_model.encode([query])[0]
                    self.embedding_cache[query_key] = query_embedding
            else:
                query_embedding = self.embedding_model.encode([query])[0]
            content_embeddings = self.compute_embeddings_batch(texts)
            similarities = self.calculate_similarity(query_embedding, content_embeddings)
            for i, item in enumerate(processed_data):
                item['similarity_score'] = float(similarities[i])
            filtered_data = []
            for item in processed_data:
                similarity_ok = item['similarity_score'] > self.similarity_threshold
                plot_relevance_ok = item.get('plot_relevance_score', 0) > self.plot_relevance_threshold
                combined_score = (item['similarity_score'] * 0.6) + (item.get('plot_relevance_score', 0) / 100 * 0.4)
                item['combined_score'] = combined_score
                if similarity_ok and plot_relevance_ok:
                    filtered_data.append(item)
            filtered_data.sort(key=lambda x: x['combined_score'], reverse=True)
            output = {
                "query": query,
                "drama_name": drama_name,
                "content_type": content_type,
                "source": "Reddit",
                "similarity_threshold": self.similarity_threshold,
                "plot_relevance_threshold": self.plot_relevance_threshold,
                "total_content_found": len(processed_data),
                "relevant_content_count": len(filtered_data),
                "data": filtered_data
            }
            self.save_data(drama_name, output)
            return output
        except Exception as e:
            logger.error(f"Search and preprocess failed: {e}")
            return {}

    def _search_with_query(self, search_query, drama_name, content_type, max_posts):
        """Internal method to search with a specific query"""
        try:
            relevant_subs = RELEVANT_SUBREDDITS.get(content_type, []) + RELEVANT_SUBREDDITS['general']
            subreddit_query = ' OR '.join([f'site:reddit.com/r/{sub}' for sub in relevant_subs[:5]])
            full_search_query = f"{search_query} ({subreddit_query})"
            response = requests.post(
                "https://google.serper.dev/search", 
                headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
                data=json.dumps({"q": full_search_query, "num": 15})
            )
            search_results = response.json()
            processed_data = []
            posts_processed = 0
            if 'organic' in search_results:
                for result in search_results['organic']:
                    if posts_processed >= max_posts:
                        break
                    if 'reddit.com' in result.get('link', ''):
                        reddit_content = get_reddit_content_enhanced(result['link'], self.scraper, content_type)
                        if reddit_content:
                            post_text = f"{reddit_content['title']} {reddit_content['post_content']}".strip()
                            if cleaned_post := clean_and_filter_text_enhanced(post_text):
                                processed_data.append({
                                    "content_id": f"post_{posts_processed + 1}",
                                    "text": cleaned_post,
                                    "plot_relevance_score": reddit_content.get('plot_relevance_score', 0),
                                    "metadata": {
                                        "type": "post", 
                                        "score": reddit_content['post_score'],
                                        "author": reddit_content['author'],
                                        "timestamp": datetime.now().isoformat()
                                    }
                                })
                            for comment in reddit_content.get('comments', []):
                                if cleaned_comment := clean_and_filter_text_enhanced(comment['body']):
                                    processed_data.append({
                                        "content_id": f"post_{posts_processed + 1}_comment_{comment['comment_number']}",
                                        "text": cleaned_comment,
                                        "plot_relevance_score": comment.get('plot_relevance_score', 0),
                                        "metadata": {
                                            "type": "comment", 
                                            "score": comment['score'],
                                            "author": comment['author'],
                                            "timestamp": datetime.now().isoformat()
                                        }
                                    })
                            posts_processed += 1
                        time.sleep(self.delay)
            return processed_data
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def save_data(self, drama_name, data):
        """Save preprocessed data with embeddings"""
        safe_name = re.sub(r'[^\w\s-]', '', drama_name).replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = config["reddit"]["scraper"]["output_file"].replace(".jsonl", f"_{safe_name}_{timestamp}.jsonl")
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Reddit data to {output_path}")

if __name__ == "__main__":
    bot = SmartRedditScraperBotWithEmbeddings()
    queries = ["explain the ending of squid game"]
    for query in queries:
        logger.info(f"Processing query: '{query}'")
        result = bot.search_and_preprocess_with_embeddings(query, num_posts=config["reddit"]["scraper"]["max_posts"])