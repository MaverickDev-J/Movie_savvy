import json
import os
import re
import uuid
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration from rag_config.yaml
try:
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "rag_config.yaml")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load rag_config.yaml: {e}")
    sys.exit(1)

# Define paths from config
RAW_DATA_DIR = config['data']['raw_dir']
PROCESSED_DATA_DIR = config['data']['processed_dir']
OUTPUT_CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, config['data']['preprocessing']['output_file'])
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, config['data']['preprocessing']['metadata_file'])
STATS_FILE = os.path.join(PROCESSED_DATA_DIR, "processing_stats.json")
CHUNK_SIZE = config['data']['preprocessing']['chunk_size']
CHUNK_OVERLAP = config['data']['preprocessing']['chunk_overlap']

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

class DataProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_chunks = []
        self.metadata = []
        self.stats = defaultdict(int)
        self.chunk_hashes = set()  # For deduplication
        
    def safe_get(self, data: Dict, key: str, default: Any = "") -> Any:
        """Safely get value from dictionary with fallback."""
        try:
            return data.get(key, default) if isinstance(data, dict) else default
        except Exception as e:
            logger.warning(f"Error accessing key '{key}': {e}")
            return default
    
    def clean_text(self, text: Any) -> str:
        """Enhanced text cleaning that preserves important formatting."""
        if not text or (isinstance(text, str) and not text.strip()):
            return "No content available"
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML tags but preserve structure
        text = re.sub(r"<[^>]+>", " ", text)
        
        # Remove excessive whitespace while preserving sentence structure
        text = re.sub(r"\s+", " ", text)
        
        # Remove only truly problematic characters, keep useful punctuation
        text = re.sub(r"[^\w\s.,!?;:()\[\]\"'-]", "", text)
        
        # Clean up multiple punctuation
        text = re.sub(r"([.!?])\1+", r"\1", text)
        
        # Trim whitespace
        text = text.strip()
        
        return text if text else "No content available"
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        if not text or text == "No content available":
            return []
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text_with_overlap(self, text: str, max_words: int = None, overlap_words: int = None) -> List[str]:
        """Split text into overlapping chunks while respecting sentence boundaries."""
        max_words = max_words or self.chunk_size
        overlap_words = overlap_words or self.chunk_overlap
        
        if not text or text == "No content available":
            return []
        
        sentences = self.split_into_sentences(text)
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max_words, finalize current chunk
            if current_word_count + sentence_words > max_words and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_word_count = 0
                
                # Add sentences from the end for overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    sent_words = len(current_chunk[i].split())
                    if overlap_word_count + sent_words <= overlap_words:
                        overlap_sentences.insert(0, current_chunk[i])
                        overlap_word_count += sent_words
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_word_count = overlap_word_count
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk if exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]
    
    def generate_chunk_hash(self, text: str) -> str:
        """Generate hash for chunk deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def is_duplicate_chunk(self, text: str) -> bool:
        """Check if chunk is duplicate."""
        chunk_hash = self.generate_chunk_hash(text)
        if chunk_hash in self.chunk_hashes:
            return True
        self.chunk_hashes.add(chunk_hash)
        return False
    
    def create_standardized_metadata(self, base_metadata: Dict) -> Dict:
        """Create standardized metadata structure."""
        return {
            "source": base_metadata.get("source", ""),
            "item_id": base_metadata.get("item_id", ""),
            "type": base_metadata.get("type", ""),
            "title": base_metadata.get("title", "Unknown"),
            "genres": base_metadata.get("genres", []),
            "rating": base_metadata.get("rating", ""),
            "score": base_metadata.get("score", ""),
            "popularity": base_metadata.get("popularity", ""),
            "tags": base_metadata.get("tags", []),
            "themes": base_metadata.get("themes", []),
            "url": base_metadata.get("url", ""),
            "additional_info": {k: v for k, v in base_metadata.items() 
                             if k not in ["source", "item_id", "type", "title", "genres", 
                                        "rating", "score", "popularity", "tags", "themes", "url"]}
        }
    
    def add_chunk(self, text: str, metadata: Dict) -> None:
        """Add a chunk with deduplication and validation."""
        if not text or text == "No content available":
            self.stats["empty_chunks"] += 1
            return
        
        if self.is_duplicate_chunk(text):
            self.stats["duplicate_chunks"] += 1
            return
        
        chunk_id = str(uuid.uuid4())
        standardized_metadata = self.create_standardized_metadata(metadata)
        
        chunk = {
            "id": chunk_id,
            "text": text,
            "metadata": standardized_metadata
        }
        
        self.processed_chunks.append(chunk)
        self.stats["valid_chunks"] += 1
    
    def process_anime(self, data: Dict, source_file: str) -> None:
        """Process anime JSON data with improved field mapping."""
        try:
            item_id = str(uuid.uuid4())
            
            # Map fields correctly based on actual JSON structure
            title = self.safe_get(data, "English Title") or self.safe_get(data, "Japanese Title", "Unknown")
            description = self.clean_text(self.safe_get(data, "Synopsis"))  # Changed from Description
            genres = self.safe_get(data, "Genres", [])
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(",") if g.strip()]
            
            themes = self.safe_get(data, "Theme", [])
            score = self.safe_get(data, "Score", "")  # Direct from root level
            popularity = self.safe_get(data, "Popularity", "")
            rating = self.safe_get(data, "Rating", "")
            
            base_metadata = {
                "source": source_file,
                "item_id": item_id,
                "title": title,
                "genres": genres,
                "themes": themes,
                "score": score,
                "popularity": popularity,
                "rating": rating,
                "url": self.safe_get(data, "Anime URL"),
                "studio": self.safe_get(data, "Studio"),
                "demographic": self.safe_get(data, "Demographic"),
                "episodes": self.safe_get(data, "Episodes"),
                "status": self.safe_get(data, "Status")
            }
            
            # Process description
            if description != "No content available":
                for chunk in self.chunk_text_with_overlap(description):
                    metadata = {**base_metadata, "type": "description"}
                    self.add_chunk(chunk, metadata)
            
            # Process reviews
            reviews_data = self.safe_get(data, "Reviews", {})
            if isinstance(reviews_data, dict):
                for review_type in ["Recommended", "Mixed Feelings", "Not Recommended"]:
                    reviews = self.safe_get(reviews_data, review_type, [])
                    if isinstance(reviews, str):
                        reviews = [reviews]
                    
                    for review in reviews:
                        if isinstance(review, dict):
                            review_text = self.clean_text(self.safe_get(review, "Review"))
                        else:
                            review_text = self.clean_text(review)
                        
                        if review_text != "No content available":
                            for chunk in self.chunk_text_with_overlap(review_text):
                                metadata = {**base_metadata, "type": f"review_{review_type.lower().replace(' ', '_')}"}
                                self.add_chunk(chunk, metadata)
            
            # Process recommendations
            recommendations = self.safe_get(data, "Recommendations", [])
            for rec in recommendations:
                if isinstance(rec, dict):
                    rec_texts = self.safe_get(rec, "Recommendation Texts", [])
                    for rec_text in rec_texts:
                        cleaned_text = self.clean_text(rec_text)
                        if cleaned_text != "No content available":
                            for chunk in self.chunk_text_with_overlap(cleaned_text):
                                metadata = {**base_metadata, "type": "recommendation"}
                                self.add_chunk(chunk, metadata)
            
            # Add to metadata
            self.metadata.append({
                "item_id": item_id,
                "source": source_file,
                "title": title,
                "genres": genres,
                "themes": themes,
                "score": score,
                "popularity": popularity,
                "rating": rating,
                "url": self.safe_get(data, "Anime URL")
            })
            
            self.stats["anime_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing anime data: {e}")
            self.stats["anime_errors"] += 1
    
    def process_kdrama_kmovie(self, data: Dict, source_file: str) -> None:
        """Process K-drama or K-movie JSON data."""
        try:
            item_id = str(uuid.uuid4())
            
            title = self.safe_get(data, "title", "Unknown")
            synopsis = self.clean_text(self.safe_get(data, "synopsis"))
            genres = self.safe_get(data, "genres", [])
            tags = self.safe_get(data, "tags", [])
            
            drama_info = self.safe_get(data, "drama_info", {})
            rating = self.safe_get(drama_info, "rating", "")
            
            base_metadata = {
                "source": source_file,
                "item_id": item_id,
                "title": title,
                "genres": genres,
                "tags": tags,
                "rating": rating,
                "url": self.safe_get(data, "link"),
                "director": self.safe_get(data, "director"),
                "screenwriter": self.safe_get(data, "screenwriter")
            }
            
            # Process synopsis
            if synopsis != "No content available":
                for chunk in self.chunk_text_with_overlap(synopsis):
                    metadata = {**base_metadata, "type": "synopsis"}
                    self.add_chunk(chunk, metadata)
            
            # Process reviews
            reviews_data = self.safe_get(data, "reviews", {})
            for review_type in ["completed", "dropped"]:
                reviews = self.safe_get(reviews_data, review_type, [])
                for review in reviews:
                    if isinstance(review, dict):
                        review_text = self.clean_text(self.safe_get(review, "text"))
                    else:
                        review_text = self.clean_text(review)
                    
                    if review_text != "No content available":
                        for chunk in self.chunk_text_with_overlap(review_text):
                            metadata = {**base_metadata, "type": f"review_{review_type}"}
                            self.add_chunk(chunk, metadata)
            
            # Process recommendations
            recommendations = self.safe_get(data, "recommendations", [])
            for rec in recommendations:
                rec_text = self.clean_text(self.safe_get(rec, "description"))
                if rec_text != "No content available":
                    for chunk in self.chunk_text_with_overlap(rec_text):
                        metadata = {**base_metadata, "type": "recommendation"}
                        self.add_chunk(chunk, metadata)
            
            # Add to metadata
            self.metadata.append({
                "item_id": item_id,
                "source": source_file,
                "title": title,
                "genres": genres,
                "tags": tags,
                "rating": rating,
                "url": self.safe_get(data, "link")
            })
            
            content_type = "kdrama" if "kdrama" in source_file else "kmovie"
            self.stats[f"{content_type}_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing K-drama/K-movie data: {e}")
            self.stats["kdrama_kmovie_errors"] += 1
    
    def process_manga(self, data: Dict, source_file: str) -> None:
        """Process manga JSON data."""
        try:
            item_id = str(uuid.uuid4())
            
            title = self.safe_get(data, "Title", "Unknown")
            description = self.clean_text(self.safe_get(data, "Description"))
            genres = self.safe_get(data, "Genres", [])
            themes = self.safe_get(data, "Theme", [])
            
            statistics = self.safe_get(data, "Statistics", {})
            score = self.safe_get(statistics, "Score", "")
            popularity = self.safe_get(statistics, "Popularity", "")
            
            base_metadata = {
                "source": source_file,
                "item_id": item_id,
                "title": title,
                "genres": genres,
                "themes": themes,
                "score": score,
                "popularity": popularity,
                "url": self.safe_get(data, "URL"),
                "authors": self.safe_get(data, "Authors", []),
                "status": self.safe_get(data, "Status"),
                "volumes": self.safe_get(data, "Volumes"),
                "chapters": self.safe_get(data, "Chapters")
            }
            
            # Process description
            if description != "No content available":
                for chunk in self.chunk_text_with_overlap(description):
                    metadata = {**base_metadata, "type": "description"}
                    self.add_chunk(chunk, metadata)
            
            # Process reviews
            reviews_data = self.safe_get(data, "Reviews", {})
            for review_type in ["Recommended", "Mixed Feelings", "Not Recommended"]:
                reviews = self.safe_get(reviews_data, review_type, [])
                for review in reviews:
                    if isinstance(review, dict):
                        review_text = self.clean_text(self.safe_get(review, "Review"))
                    else:
                        review_text = self.clean_text(review)
                    
                    if review_text != "No content available":
                        for chunk in self.chunk_text_with_overlap(review_text):
                            metadata = {**base_metadata, "type": f"review_{review_type.lower().replace(' ', '_')}"}
                            self.add_chunk(chunk, metadata)
            
            # Process recommendations
            recommendations = self.safe_get(data, "Recommendations", [])
            for rec in recommendations:
                rec_texts = self.safe_get(rec, "Recommendation Texts", [])
                for rec_text in rec_texts:
                    cleaned_text = self.clean_text(rec_text)
                    if cleaned_text != "No content available":
                        for chunk in self.chunk_text_with_overlap(cleaned_text):
                            metadata = {**base_metadata, "type": "recommendation"}
                            self.add_chunk(chunk, metadata)
            
            # Add to metadata
            self.metadata.append({
                "item_id": item_id,
                "source": source_file,
                "title": title,
                "genres": genres,
                "themes": themes,
                "score": score,
                "popularity": popularity,
                "url": self.safe_get(data, "URL")
            })
            
            self.stats["manga_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing manga data: {e}")
            self.stats["manga_errors"] += 1
    
    def process_hollywood_bollywood(self, data: Dict, source_file: str) -> None:
        """Process Hollywood or Bollywood JSON data."""
        try:
            item_id = str(uuid.uuid4())
            
            title = self.safe_get(data, "Title", "Unknown")
            plot = self.clean_text(self.safe_get(data, "Plot"))
            genres = self.safe_get(data, "Genres", [])
            rating = self.safe_get(data, "Rating", "")
            
            base_metadata = {
                "source": source_file,
                "item_id": item_id,
                "title": title,
                "genres": genres,
                "rating": rating,
                "url": self.safe_get(data, "URL"),
                "director": self.safe_get(data, "Director"),
                "year": self.safe_get(data, "Year"),
                "runtime": self.safe_get(data, "Runtime")
            }
            
            # Process plot
            if plot != "No content available":
                for chunk in self.chunk_text_with_overlap(plot):
                    metadata = {**base_metadata, "type": "plot"}
                    self.add_chunk(chunk, metadata)
            
            # Process plot summaries
            plot_summaries = self.safe_get(data, "PlotSummaries", [])
            for summary in plot_summaries:
                summary_text = self.clean_text(self.safe_get(summary, "Content"))
                if summary_text != "No content available":
                    for chunk in self.chunk_text_with_overlap(summary_text):
                        metadata = {**base_metadata, "type": "plot_summary"}
                        self.add_chunk(chunk, metadata)
            
            # Process reviews
            reviews = self.safe_get(data, "Reviews", [])
            for review in reviews:
                review_text = self.clean_text(self.safe_get(review, "Content"))
                review_rating = self.safe_get(review, "Rating", 0)
                if review_text != "No content available":
                    for chunk in self.chunk_text_with_overlap(review_text):
                        metadata = {**base_metadata, "type": "review", "review_rating": review_rating}
                        self.add_chunk(chunk, metadata)
            
            # Process trivia
            trivia_list = self.safe_get(data, "Trivia", [])
            for trivia in trivia_list:
                trivia_text = self.clean_text(trivia)
                if trivia_text != "No content available":
                    for chunk in self.chunk_text_with_overlap(trivia_text):
                        metadata = {**base_metadata, "type": "trivia"}
                        self.add_chunk(chunk, metadata)
            
            # Add to metadata
            self.metadata.append({
                "item_id": item_id,
                "source": source_file,
                "title": title,
                "genres": genres,
                "rating": rating,
                "url": self.safe_get(data, "URL")
            })
            
            content_type = "hollywood" if "hollywood" in source_file else "bollywood"
            self.stats[f"{content_type}_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing Hollywood/Bollywood data: {e}")
            self.stats["hollywood_bollywood_errors"] += 1
    
    def validate_chunk_quality(self) -> None:
        """Validate processed chunks for quality."""
        valid_chunks = []
        
        for chunk in self.processed_chunks:
            text = chunk.get("text", "")
            
            # Skip very short chunks
            if len(text.split()) < 5:
                self.stats["chunks_too_short"] += 1
                continue
            
            # Skip chunks with too many repeated words
            words = text.lower().split()
            if len(set(words)) / len(words) < 0.3:  # Less than 30% unique words
                self.stats["chunks_low_diversity"] += 1
                continue
            
            valid_chunks.append(chunk)
        
        self.processed_chunks = valid_chunks
        self.stats["final_valid_chunks"] = len(valid_chunks)
    
    def process_file(self, file_path: str, file_name: str) -> None:
        """Process a single JSON file."""
        try:
            logger.info(f"Processing {file_name}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both single objects and arrays
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                if file_name == "anime.json":
                    self.process_anime(item, file_name)
                elif file_name in ["kdrama.json", "kmovie.json"]:
                    self.process_kdrama_kmovie(item, file_name)
                elif file_name == "manga.json":
                    self.process_manga(item, file_name)
                elif file_name in ["hollywood.json", "bollywood.json"]:
                    self.process_hollywood_bollywood(item, file_name)
                else:
                    logger.warning(f"Unknown file type: {file_name}")
            
            logger.info(f"Completed processing {file_name}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_name}: {e}")
            self.stats["json_errors"] += 1
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            self.stats["file_errors"] += 1
    
    def save_results(self) -> None:
        """Save processed chunks, metadata, and statistics."""
        try:
            # Save chunks
            with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
                for chunk in self.processed_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
            # Save metadata
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            # Save statistics
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(dict(self.stats), f, indent=2)
            
            logger.info(f"Saved {len(self.processed_chunks)} chunks to {OUTPUT_CHUNKS_FILE}")
            logger.info(f"Saved {len(self.metadata)} metadata entries to {METADATA_FILE}")
            logger.info(f"Saved processing statistics to {STATS_FILE}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            sys.exit(1)
    
    def print_statistics(self) -> None:
        """Print processing statistics."""
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        
        for key, value in sorted(self.stats.items()):
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("="*50)

def main():
    """Main function to process all JSON files."""
    try:
        processor = DataProcessor()
        
        if not os.path.exists(RAW_DATA_DIR):
            logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
            sys.exit(1)
        
        json_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".json")]
        
        if not json_files:
            logger.warning(f"No JSON files found in {RAW_DATA_DIR}")
            sys.exit(1)
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        for file_name in json_files:
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            processor.process_file(file_path, file_name)
        
        # Validate chunk quality
        logger.info("Validating chunk quality...")
        processor.validate_chunk_quality()
        
        # Save results
        processor.save_results()
        
        # Print statistics
        processor.print_statistics()
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Main processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()