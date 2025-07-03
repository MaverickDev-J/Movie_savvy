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

# Expanded subreddits for better coverage
RELEVANT_SUBREDDITS = {
    'anime': ['anime', 'animesuggest', 'animediscussion', 'manga', 'onepiece', 'naruto', 'attackontitan', 'dragonball', 'demonslayer', 'jujutsukaisen', 'myheroacademia'],
    'kdrama': ['kdrama', 'koreandramas', 'kdramas', 'asiandrama', 'kdramarecommends'],
    'kmovie': ['koreanmovies', 'asianmovies', 'worldcinema', 'koreanfilm'],
    'bollywood': ['bollywood', 'indianmovies', 'bollywoodmemes', 'hindicinema', 'tollywood'],
    'hollywood': ['movies', 'moviesuggestions', 'filmmakers', 'moviedetails', 'marvelstudios', 'dc_cinematic', 'horror', 'scifi'],
    'tv_series': ['television', 'tvshows', 'netflix', 'amazonprime', 'hbo', 'disney', 'appletv'],
    'gaming': ['gaming', 'games', 'nintendo', 'playstation', 'xbox', 'pcgaming', 'mobilegaming'],
    'books': ['books', 'fantasy', 'scifi', 'literature', 'booksuggestions'],
    'music': ['music', 'hiphop', 'rock', 'pop', 'kpop', 'jpop'],
    'sports': ['sports', 'soccer', 'football', 'basketball', 'baseball', 'cricket'],
    'technology': ['technology', 'programming', 'artificial', 'machinelearning', 'datascience'],
    'general': ['movies', 'television', 'entertainment', 'popculture', 'askreddit', 'explainlikeimfive', 'nostupidquestions']
}

# Universal content relevance keywords - much broader
CONTENT_RELEVANCE_KEYWORDS = {
    'story_elements': ['plot', 'story', 'storyline', 'narrative', 'character', 'protagonist', 'antagonist', 'backstory', 'lore', 'history'],
    'discussion_terms': ['what', 'who', 'why', 'how', 'when', 'where', 'explain', 'explanation', 'theory', 'analysis', 'interpretation', 'meaning', 'about', 'regarding'],
    'descriptive_terms': ['describe', 'tell', 'details', 'information', 'facts', 'truth', 'reality', 'actually', 'really', 'truly'],
    'question_indicators': ['question', 'ask', 'wondering', 'curious', 'confused', 'unclear', 'understand', 'know', 'find out'],
    'content_types': ['episode', 'season', 'chapter', 'volume', 'part', 'series', 'movie', 'film', 'show', 'book', 'game', 'album', 'song'],
    'emotional_indicators': ['love', 'hate', 'like', 'dislike', 'favorite', 'best', 'worst', 'amazing', 'terrible', 'good', 'bad', 'great'],
    'comparative_terms': ['vs', 'versus', 'compare', 'comparison', 'similar', 'different', 'better', 'worse', 'same', 'unlike'],
    'temporal_terms': ['before', 'after', 'during', 'while', 'then', 'now', 'later', 'earlier', 'recent', 'latest', 'new', 'old']
}

# Reduced promotional keywords - less restrictive
PROMOTIONAL_KEYWORDS = [
    'buy now', 'purchase', 'sale', 'discount', 'price', 'cost', 'cheap', 'expensive',
    'advertisement', 'sponsor', 'promoted', 'affiliate', 'referral'
]

def extract_drama_name_and_type(query):
    """Enhanced extraction for any type of query"""
    query_lower = query.lower().strip()
    
    # Enhanced content type detection
    content_types = {
        'anime': ['anime', 'manga', 'one piece', 'naruto', 'dragon ball', 'attack on titan', 'demon slayer', 'jujutsu kaisen'],
        'kdrama': ['kdrama', 'korean drama', 'k-drama', 'korean series'],
        'kmovie': ['kmovie', 'korean movie', 'k-movie', 'korean film'],
        'bollywood': ['bollywood', 'hindi movie', 'hindi film', 'indian movie'],
        'hollywood': ['hollywood', 'american movie', 'american film', 'marvel', 'dc comics'],
        'tv_series': ['tv series', 'tv show', 'series', 'show', 'netflix', 'hbo', 'disney'],
        'gaming': ['game', 'gaming', 'video game', 'nintendo', 'playstation', 'xbox'],
        'books': ['book', 'novel', 'author', 'literature'],
        'music': ['song', 'album', 'artist', 'band', 'music', 'singer'],
        'sports': ['sport', 'team', 'player', 'match', 'game', 'league'],
        'technology': ['tech', 'software', 'app', 'program', 'code', 'ai', 'ml']
    }
    
    detected_type = 'general'
    for content_type, keywords in content_types.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected_type = content_type
                break
        if detected_type != 'general':
            break
    
    # More flexible name extraction patterns
    patterns = [
        # Character questions: "who is X in Y"
        r'who\s+is\s+([a-zA-Z0-9\s\-\'\":]+?)\s+in\s+([a-zA-Z0-9\s\-\'\":]+?)(?:\s+reddit)?$',
        # What questions: "what is X about"
        r'what\s+is\s+([a-zA-Z0-9\s\-\'\":]+?)\s+about',
        # Explain questions: "explain X"
        r'explain\s+([a-zA-Z0-9\s\-\'\":]+?)(?:\s+reddit)?$',
        # General questions about something
        r'(?:about|regarding)\s+([a-zA-Z0-9\s\-\'\":]+?)(?:\s+reddit)?$',
        # Direct title mentions
        r'^([a-zA-Z0-9\s\-\'\":]+?)\s+(?:question|discussion|theory|analysis|review|spoiler)s?$',
        # Fallback - just extract meaningful words
        r'^(.+?)(?:\s+reddit)?$'
    ]
    
    drama_name = None
    for pattern in patterns:
        if match := re.search(pattern, query_lower, re.IGNORECASE):
            if len(match.groups()) == 2:  # Character in series pattern
                character_name = match.group(1).strip()
                series_name = match.group(2).strip()
                drama_name = f"{character_name} in {series_name}".title()
            else:
                extracted = match.group(1).strip()
                # Clean up the extracted name
                extracted = re.sub(r'^(the\s+)', '', extracted, flags=re.IGNORECASE)
                extracted = re.sub(r'\s+(movie|film|drama|series|show|anime|manga|book|game)s?$', '', extracted, flags=re.IGNORECASE)
                extracted = re.sub(r'[^\w\s\-\'\":.]', '', extracted).strip()
                
                # Filter out stop words but be less aggressive
                stop_words = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                             'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                             'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                
                words = extracted.split()
                meaningful_words = [w for w in words if w.lower() not in stop_words and len(w) > 1]
                
                if meaningful_words and len(' '.join(meaningful_words)) > 1:
                    drama_name = ' '.join(meaningful_words).title()
                    break
    
    # If still no drama name, use all meaningful words from original query
    if not drama_name:
        original_words = re.findall(r'\b[a-zA-Z0-9]+\b', query)
        stop_words = {'is', 'are', 'was', 'were', 'what', 'who', 'where', 'when', 'why', 'how', 
                     'reddit', 'discussion', 'question', 'about', 'regarding', 'concerning'}
        meaningful_words = [w for w in original_words if w.lower() not in stop_words and len(w) > 1]
        drama_name = ' '.join(meaningful_words[:6]).title() if meaningful_words else "General Discussion"
    
    return drama_name, detected_type

def calculate_content_relevance_score(text, query):
    """Calculate how relevant the text is to the query - much more flexible"""
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Direct query word matching (most important)
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    common_words = query_words.intersection(text_words)
    query_match_score = (len(common_words) / len(query_words)) * 100 if query_words else 0
    
    # Content relevance keywords
    content_score = 0
    total_keywords = 0
    for category, keywords in CONTENT_RELEVANCE_KEYWORDS.items():
        category_score = 0
        for keyword in keywords:
            if keyword in text_lower:
                category_score += 1
        # Weight discussion terms higher as they indicate engagement
        if category == 'discussion_terms':
            content_score += category_score * 2
        else:
            content_score += category_score
        total_keywords += len(keywords)
    
    # Promotional content penalty (reduced)
    promotional_penalty = 0
    for keyword in PROMOTIONAL_KEYWORDS:
        if keyword in text_lower:
            promotional_penalty += 1
    
    # Calculate base score
    keyword_relevance = min((content_score / max(total_keywords * 0.05, 1)) * 100, 100)
    
    # Combine scores with query matching being most important
    final_score = (query_match_score * 0.6) + (keyword_relevance * 0.4) - (promotional_penalty * 5)
    
    return max(0, min(100, final_score))

def is_relevant_subreddit(url, content_type):
    """Enhanced subreddit relevance check"""
    try:
        subreddit_match = re.search(r'reddit\.com/r/([^/]+)', url)
        if not subreddit_match:
            return False
        
        subreddit = subreddit_match.group(1).lower()
        
        # Get relevant subreddits for the content type
        relevant_subs = RELEVANT_SUBREDDITS.get(content_type, []) + RELEVANT_SUBREDDITS['general']
        
        # More flexible matching
        return any(sub in subreddit or subreddit in sub for sub in relevant_subs)
    except:
        return False

def is_quality_content(text, query, min_words=10, max_words=800, min_relevance_score=10):
    """Enhanced quality check that's more flexible"""
    if not text or len(text.strip()) < 5:
        return False
    
    words = text.split()
    word_count = len(words)
    
    # More flexible word count requirements
    if word_count < min_words or word_count > max_words:
        return False
    
    # Calculate relevance to the specific query
    relevance_score = calculate_content_relevance_score(text, query)
    if relevance_score < min_relevance_score:
        return False
    
    # Less restrictive meme patterns
    meme_patterns = [
        r'^(lol|lmao|haha|omg|wow)\.?$',
        r'^(\+1|this|same|exactly)\.?$',
        r'^(first|second|third)\.?$'
    ]
    
    text_lower = text.lower().strip()
    if any(re.match(pattern, text_lower) for pattern in meme_patterns):
        return False
    
    # Content substance indicators
    substance_indicators = ['because', 'however', 'although', 'when', 'where', 'why', 'how', 
                          'what', 'which', 'who', 'actually', 'really', 'think', 'believe',
                          'according', 'mentioned', 'explained', 'described', 'discussed']
    
    has_substance = any(indicator in text_lower for indicator in substance_indicators)
    
    return has_substance or word_count > 20

def is_bot_account(author_name):
    """Check if account is likely a bot"""
    if not author_name or author_name in ['[deleted]', '[removed]', 'AutoModerator']:
        return True
    
    bot_indicators = ['bot', 'auto', 'moderator', 'reminder', 'converter']
    return any(indicator in author_name.lower() for indicator in bot_indicators)

def get_reddit_content_enhanced(url, scraper, content_type, query):
    """Get Reddit content with enhanced filtering for any query"""
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
            
            # More lenient scoring requirements
            if submission.score < 2 or is_bot_account(submission.author.name if submission.author else None):
                return None
            
            post_text = f"{submission.title} {submission.selftext}".strip()
            
            # Use query-specific quality check
            if not is_quality_content(post_text, query, min_words=5, max_words=1500, min_relevance_score=5):
                return None
            
            content = {
                "title": submission.title,
                "post_content": submission.selftext or "No text content",
                "post_score": submission.score,
                "author": submission.author.name if submission.author else "[deleted]",
                "content_relevance_score": calculate_content_relevance_score(post_text, query),
                "comments": []
            }
            
            submission.comments.replace_more(limit=0)
            all_comments = [c for c in submission.comments.list() if hasattr(c, 'body') and c.body != '[deleted]']
            
            # Filter quality comments
            quality_comments = []
            for comment in all_comments:
                if (comment.score >= 1 and 
                    not is_bot_account(comment.author.name if comment.author else None) and
                    is_quality_content(comment.body, query, min_relevance_score=10)):
                    quality_comments.append(comment)
            
            # Score comments based on relevance and engagement
            def comment_score(c):
                relevance_score = calculate_content_relevance_score(c.body, query)
                return (c.score * 0.3) + (relevance_score * 0.7)
            
            top_comments = sorted(quality_comments, key=comment_score, reverse=True)[:2]
            
            for idx, comment in enumerate(top_comments, 1):
                content["comments"].append({
                    "comment_number": idx,
                    "body": comment.body,
                    "score": comment.score,
                    "author": comment.author.name if comment.author else "[deleted]",
                    "content_relevance_score": calculate_content_relevance_score(comment.body, query)
                })
            
            return content
            
    except Exception as e:
        logger.error(f"Reddit API failed for {url}: {e}")
        return None

def clean_and_filter_text_enhanced(text, query):
    """Enhanced text cleaning and filtering"""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return None
    
    # Clean up text
    text = re.sub(r'#\s*SPOILER ALERT\s*|\n+|\*\w+\*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Use query-specific quality check
    if not is_quality_content(text, query, min_relevance_score=10):
        return None
    
    return text

class SmartRedditScraperBotWithEmbeddings:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.delay = 2
        self.output_dir = str(BASE_DIR / config["data"]["reddit_dir"])
        self.similarity_threshold = 0.4  # Lowered for more flexibility
        self.relevance_threshold = 15  # Lowered for more flexibility
        
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
        """Create fallback search queries for better results - more diverse"""
        fallback_queries = []
        
        # Extract key terms from query
        query_words = re.findall(r'\b\w+\b', query.lower())
        key_words = [w for w in query_words if len(w) > 2 and w not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'reddit']]
        
        # Create various search combinations
        if len(key_words) >= 2:
            fallback_queries.append(f"{' '.join(key_words[:3])} reddit discussion")
            fallback_queries.append(f"{' '.join(key_words[:2])} reddit")
            fallback_queries.append(f"{key_words[0]} {key_words[-1]} reddit")
        
        # Add content-type specific searches
        if content_type != 'general':
            fallback_queries.append(f"{drama_name.lower()} reddit {content_type}")
        
        return fallback_queries[:4]

    def search_and_preprocess_with_embeddings(self, query, num_posts):
        """Search Reddit, preprocess, and calculate embeddings with universal query support"""
        drama_name, content_type = extract_drama_name_and_type(query)
        logger.info(f"Detected: '{drama_name}' ({content_type}) | Searching...")
        
        if not SERPER_API_KEY:
            logger.error("No SERPER API key set")
            return {}
        
        try:
            processed_data = []
            
            # Primary search
            processed_data.extend(self._search_with_query(query, drama_name, content_type, num_posts, query))
            
            # Fallback searches if needed
            if len(processed_data) < num_posts:
                logger.info(f"Primary search yielded {len(processed_data)} results. Trying fallback searches...")
                fallback_queries = self.create_fallback_searches(query, drama_name, content_type)
                
                for fallback_query in fallback_queries:
                    if len(processed_data) >= num_posts:
                        break
                    logger.info(f"Fallback search: '{fallback_query}'")
                    fallback_data = self._search_with_query(fallback_query, drama_name, content_type, 
                                                          num_posts - len(processed_data), query)
                    processed_data.extend(fallback_data)
            
            if not processed_data:
                logger.warning("No quality content found")
                return {}
            
            # Calculate embeddings and similarities
            texts = [item['text'] for item in processed_data]
            
            # Handle query embedding with cache
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
            
            # Add similarity scores
            for i, item in enumerate(processed_data):
                item['similarity_score'] = float(similarities[i])
            
            # Filter and rank results
            filtered_data = []
            for item in processed_data:
                similarity_ok = item['similarity_score'] > self.similarity_threshold
                relevance_ok = item.get('content_relevance_score', 0) > self.relevance_threshold
                
                # Combined score weighting both similarity and relevance
                combined_score = (item['similarity_score'] * 0.5) + (item.get('content_relevance_score', 0) / 100 * 0.5)
                item['combined_score'] = combined_score
                
                if similarity_ok or relevance_ok:  # More flexible - either condition can pass
                    filtered_data.append(item)
            
            # Sort by combined score
            filtered_data.sort(key=lambda x: x['combined_score'], reverse=True)
            
            output = {
                "query": query,
                "drama_name": drama_name,
                "content_type": content_type,
                "source": "Reddit",
                "similarity_threshold": self.similarity_threshold,
                "relevance_threshold": self.relevance_threshold,
                "total_content_found": len(processed_data),
                "relevant_content_count": len(filtered_data),
                "data": filtered_data
            }
            
            self.save_data(drama_name, output)
            return output
            
        except Exception as e:
            logger.error(f"Search and preprocess failed: {e}")
            return {}

    def _search_with_query(self, search_query, drama_name, content_type, max_posts, original_query):
        """Internal method to search with a specific query"""
        try:
            # Get relevant subreddits
            relevant_subs = RELEVANT_SUBREDDITS.get(content_type, []) + RELEVANT_SUBREDDITS['general']
            
            # Create more targeted search
            subreddit_query = ' OR '.join([f'site:reddit.com/r/{sub}' for sub in relevant_subs[:8]])
            full_search_query = f"{search_query} ({subreddit_query})"
            
            response = requests.post(
                "https://google.serper.dev/search", 
                headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
                data=json.dumps({"q": full_search_query, "num": 20})  # Increased for better results
            )
            
            search_results = response.json()
            processed_data = []
            posts_processed = 0
            
            if 'organic' in search_results:
                for result in search_results['organic']:
                    if posts_processed >= max_posts:
                        break
                    
                    if 'reddit.com' in result.get('link', ''):
                        reddit_content = get_reddit_content_enhanced(result['link'], self.scraper, content_type, original_query)
                        
                        if reddit_content:
                            post_text = f"{reddit_content['title']} {reddit_content['post_content']}".strip()
                            
                            if cleaned_post := clean_and_filter_text_enhanced(post_text, original_query):
                                processed_data.append({
                                    "content_id": f"post_{posts_processed + 1}",
                                    "text": cleaned_post,
                                    "content_relevance_score": reddit_content.get('content_relevance_score', 0),
                                    "metadata": {
                                        "type": "post", 
                                        "score": reddit_content['post_score'],
                                        "author": reddit_content['author'],
                                        "timestamp": datetime.now().isoformat()
                                    }
                                })
                            
                            # Process comments
                            for comment in reddit_content.get('comments', []):
                                if cleaned_comment := clean_and_filter_text_enhanced(comment['body'], original_query):
                                    processed_data.append({
                                        "content_id": f"post_{posts_processed + 1}_comment_{comment['comment_number']}",
                                        "text": cleaned_comment,
                                        "content_relevance_score": comment.get('content_relevance_score', 0),
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
    queries = ["who is imu in one piece reddit"]
    
    for query in queries:
        logger.info(f"Processing query: '{query}'")
        result = bot.search_and_preprocess_with_embeddings(query, num_posts=config["reddit"]["scraper"]["max_posts"])