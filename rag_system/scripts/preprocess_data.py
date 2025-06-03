import json
import os
import re
import uuid
from pathlib import Path

# Define paths
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
OUTPUT_CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, "processed_chunks.jsonl")
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.json")

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def clean_text(text):
    """Clean text by removing HTML tags, special characters, and extra whitespace."""
    if not isinstance(text, str) or not text.strip():
        return "No content available"
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    text = " ".join(text.split())
    text = text.lower()
    return text if text.strip() else "No content available"

def chunk_text(text, max_length=200):
    """Split text into chunks of approximately max_length words."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks if chunks else [text]

def process_anime(data, source_file):
    """Process anime JSON data, handling reviews as strings or dictionaries."""
    chunks = []
    metadata = []
    item_id = str(uuid.uuid4())

    description = clean_text(data.get("Description", ""))
    genres = data.get("Genres", [])
    themes = data.get("Theme", [])
    statistics = data.get("Statistics", {})
    score = statistics.get("Score", "")
    popularity = statistics.get("Popularity", "")

    if description != "No content available":
        for chunk in chunk_text(description):
            chunk_id = str(uuid.uuid4())
            chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                "source": source_file,
                "item_id": item_id,
                "type": "description",
                "genres": genres,
                "themes": themes,
                "score": score,
                "popularity": popularity
            }})

    for review_type in ["Recommended", "Mixed Feelings", "Not Recommended"]:
        for review in data.get("Reviews", {}).get(review_type, []):
            if isinstance(review, str):
                review_text = clean_text(review)
            else:
                if isinstance(review, str):
                    review_text = clean_text(review)
                else:
                    review_text = clean_text(review.get("Review", ""))
            if review_text != "No content available":
                for chunk in chunk_text(review_text):
                    chunk_id = str(uuid.uuid4())
                    chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                        "source": source_file,
                        "item_id": item_id,
                        "type": f"review_{review_type.lower().replace(' ', '_')}",
                        "genres": genres,
                        "themes": themes,
                        "score": score,
                        "popularity": popularity
                    }})

    for rec in data.get("Recommendations", []):
        for rec_text in rec.get("Recommendation Texts", []):
            rec_text = clean_text(rec_text)
            if rec_text != "No content available":
                for chunk in chunk_text(rec_text):
                    chunk_id = str(uuid.uuid4())
                    chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                        "source": source_file,
                        "item_id": item_id,
                        "type": "recommendation",
                        "genres": genres,
                        "themes": themes,
                        "score": score,
                        "popularity": popularity
                    }})

    metadata.append({
        "item_id": item_id,
        "source": source_file,
        "title": data.get("Title", "Unknown"),
        "genres": genres,
        "themes": themes,
        "score": score,
        "popularity": popularity,
        "url": data.get("URL", "")
    })
    return chunks, metadata

def process_kdrama_kmovie(data, source_file):
    """Process K-drama or K-movie JSON data."""
    chunks = []
    metadata = []
    item_id = str(uuid.uuid4())

    synopsis = clean_text(data.get("synopsis", ""))
    genres = data.get("genres", [])
    tags = data.get("tags", [])
    title = data.get("title", "Unknown")
    drama_info = data.get("drama_info", {})
    rating = drama_info.get("rating", "")  # Check for rating in drama_info

    if synopsis != "No content available":
        for chunk in chunk_text(synopsis):
            chunk_id = str(uuid.uuid4())
            chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                "source": source_file,
                "item_id": item_id,
                "type": "synopsis",
                "genres": genres,
                "tags": tags,
                "title": title,
                "rating": rating
            }})

    for review_type in ["completed", "dropped"]:
        for review in data.get("reviews", {}).get(review_type, []):
            if isinstance(review, str):
                review_text = clean_text(review)
            else:
                review_text = clean_text(review.get("text", ""))
            if review_text != "No content available":
                for chunk in chunk_text(review_text):
                    chunk_id = str(uuid.uuid4())
                    chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                        "source": source_file,
                        "item_id": item_id,
                        "type": f"review_{review_type}",
                        "genres": genres,
                        "tags": tags,
                        "title": title,
                        "rating": rating
                    }})

    for rec in data.get("recommendations", []):
        rec_text = clean_text(rec.get("description", ""))
        if rec_text != "No content available":
            for chunk in chunk_text(rec_text):
                chunk_id = str(uuid.uuid4())
                chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                    "source": source_file,
                    "item_id": item_id,
                    "type": "recommendation",
                    "genres": genres,
                    "tags": tags,
                    "title": title,
                    "rating": rating
                }})

    metadata.append({
        "item_id": item_id,
        "source": source_file,
        "title": title,
        "genres": genres,
        "tags": tags,
        "rating": rating,
        "link": data.get("link", "")
    })
    return chunks, metadata

def process_manga(data, source_file):
    """Process manga JSON data."""
    chunks = []
    metadata = []
    item_id = str(uuid.uuid4())

    # Use capitalized field names to match your JSON
    description = clean_text(data.get("Description", ""))
    genres = data.get("Genres", [])
    statistics = data.get("Statistics", {})
    score = statistics.get("Score", "")  # Capitalized
    popularity = statistics.get("Popularity", "")  # Capitalized

    # Process description chunks...
    if description != "No content available":
        for chunk in chunk_text(description):
            chunk_id = str(uuid.uuid4())
            chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                "source": source_file,
                "item_id": item_id,
                "type": "description",
                "genres": genres,
                "score": score,
                "popularity": popularity
            }})

    # Process reviews with correct structure
    for review_type in ["Recommended", "Mixed Feelings", "Not Recommended"]:
        for review in data.get("Reviews", {}).get(review_type, []):
            if isinstance(review, str):
                review_text = clean_text(review)
            else:
                review_text = clean_text(review.get("Review", ""))
            if review_text != "No content available":
                for chunk in chunk_text(review_text):
                    chunk_id = str(uuid.uuid4())
                    chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                        "source": source_file,
                        "item_id": item_id,
                        "type": f"review_{review_type.lower().replace(' ', '_')}",
                        "genres": genres,
                        "score": score,
                        "popularity": popularity
                    }})

    # Process recommendations
    for rec in data.get("Recommendations", []):
        for rec_text in rec.get("Recommendation Texts", []):
            rec_text = clean_text(rec_text)
            if rec_text != "No content available":
                for chunk in chunk_text(rec_text):
                    chunk_id = str(uuid.uuid4())
                    chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                        "source": source_file,
                        "item_id": item_id,
                        "type": "recommendation",
                        "genres": genres,
                        "score": score,
                        "popularity": popularity
                    }})

    metadata.append({
        "item_id": item_id,
        "source": source_file,
        "title": data.get("Title", "Unknown"),  # Added title
        "genres": genres,
        "score": score,
        "popularity": popularity,
        "url": data.get("URL", "")
    })
    return chunks, metadata

def process_hollywood_bollywood(data, source_file):
    """Process Hollywood or Bollywood JSON data, including Trivia."""
    chunks = []
    metadata = []
    item_id = str(uuid.uuid4())

    plot = clean_text(data.get("Plot", ""))
    genres = data.get("Genres", [])
    title = data.get("Title", "Unknown")
    rating = data.get("Rating", "")

    if plot != "No content available":
        for chunk in chunk_text(plot):
            chunk_id = str(uuid.uuid4())
            chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                "source": source_file,
                "item_id": item_id,
                "type": "plot",
                "genres": genres,
                "rating": rating,
                "title": title
            }})

    for summary in data.get("PlotSummaries", []):
        summary_text = clean_text(summary.get("Content", ""))
        if summary_text != "No content available":
            for chunk in chunk_text(summary_text):
                chunk_id = str(uuid.uuid4())
                chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                    "source": source_file,
                    "item_id": item_id,
                    "type": "plot_summary",
                    "genres": genres,
                    "rating": rating,
                    "title": title
                }})

    for review in data.get("Reviews", []):
        review_text = clean_text(review.get("Content", ""))
        review_rating = review.get("Rating", 0)
        if review_text != "No content available":
            for chunk in chunk_text(review_text):
                chunk_id = str(uuid.uuid4())
                chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                    "source": source_file,
                    "item_id": item_id,
                    "type": "review",
                    "genres": genres,
                    "rating": review_rating,
                    "title": title
                }})

    for trivia in data.get("Trivia", []):
        trivia_text = clean_text(trivia)
        if trivia_text != "No content available":
            for chunk in chunk_text(trivia_text):
                chunk_id = str(uuid.uuid4())
                chunks.append({"id": chunk_id, "text": chunk, "metadata": {
                    "source": source_file,
                    "item_id": item_id,
                    "type": "trivia",
                    "genres": genres,
                    "rating": rating,
                    "title": title
                }})

    metadata.append({
        "item_id": item_id,
        "source": source_file,
        "title": title,
        "genres": genres,
        "rating": rating,
        "url": data.get("URL", "")
    })
    return chunks, metadata

def main():
    """Main function to process all JSON files."""
    all_chunks = []
    all_metadata = []

    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".json"):
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if file_name == "anime.json":
                if isinstance(data, list):
                    for item in data:
                        chunks, metadata = process_anime(item, file_name)
                        all_chunks.extend(chunks)
                        all_metadata.extend(metadata)
                else:
                    chunks, metadata = process_anime(data, file_name)
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)
            elif file_name in ["kdrama.json", "kmovie.json"]:
                if isinstance(data, list):
                    for item in data:
                        chunks, metadata = process_kdrama_kmovie(item, file_name)
                        all_chunks.extend(chunks)
                        all_metadata.extend(metadata)
                else:
                    chunks, metadata = process_kdrama_kmovie(data, file_name)
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)
            


            elif file_name == "manga.json":
                if isinstance(data, list):
                    for item in data:
                        chunks, metadata = process_manga(item, file_name)
                        all_chunks.extend(chunks)
                        all_metadata.extend(metadata)
                else:
                    chunks, metadata = process_manga(data, file_name)  # Single object
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)

            
            elif file_name in ["hollywood.json", "bollywood.json"]:
                if isinstance(data, list):
                    for item in data:
                        chunks, metadata = process_hollywood_bollywood(item, file_name)
                        all_chunks.extend(chunks)
                        all_metadata.extend(metadata)
                else:
                    chunks, metadata = process_hollywood_bollywood(data, file_name)
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)

    with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Processed {len(all_chunks)} chunks and saved to {OUTPUT_CHUNKS_FILE}")
    print(f"Saved {len(all_metadata)} metadata entries to {METADATA_FILE}")

if __name__ == "__main__":
    main()










# import json
# import os
# import re
# import uuid
# from pathlib import Path
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Define paths
# RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
# PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
# OUTPUT_CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, "processed_chunks.jsonl")
# METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.json")

# # Ensure processed directory exists
# os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# def clean_text(text):
#     """Clean text by removing HTML tags, special characters, and extra whitespace."""
#     if not isinstance(text, str) or not text.strip():
#         logger.debug(f"Text empty or invalid: {text}")
#         return "No content available"
#     text = re.sub(r"<[^>]+>", "", text)
#     text = re.sub(r"[^\w\s.,!?-]", "", text)
#     text = " ".join(text.split())
#     text = text.lower()
#     return text if text.strip() else "No content available"

# def chunk_text(text, max_length=200):
#     """Split text into chunks of approximately max_length words."""
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_length = 0

#     for word in words:
#         current_chunk.append(word)
#         current_length += 1
#         if current_length >= max_length:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#             current_length = 0
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
#     logger.debug(f"Generated {len(chunks)} chunks for text: {text[:50]}...")
#     return chunks if chunks else [text]

# def process_anime(data, source_file):
#     """Process anime JSON data, handling reviews as strings or dictionaries."""
#     chunks = []
#     metadata = []
#     item_id = str(uuid.uuid4())

#     description = clean_text(data.get("Description", ""))
#     genres = data.get("Genres", [])
#     themes = data.get("Theme", [])
#     statistics = data.get("Statistics", {})
#     score = statistics.get("Score", "")
#     popularity = statistics.get("Popularity", "")

#     if description != "No content available":
#         for chunk in chunk_text(description):
#             chunk_id = str(uuid.uuid4())
#             chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                 "source": source_file,
#                 "item_id": item_id,
#                 "type": "description",
#                 "genres": genres,
#                 "themes": themes,
#                 "score": score,
#                 "popularity": popularity
#             }})

#     for review_type in ["Recommended", "Mixed Feelings", "Not Recommended"]:
#         for review in data.get("Reviews", {}).get(review_type, []):
#             if isinstance(review, str):
#                 review_text = clean_text(review)
#             else:
#                 review_text = clean_text(review.get("Review", ""))
#             if review_text != "No content available":
#                 for chunk in chunk_text(review_text):
#                     chunk_id = str(uuid.uuid4())
#                     chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                         "source": source_file,
#                         "item_id": item_id,
#                         "type": f"review_{review_type.lower().replace(' ', '_')}",
#                         "genres": genres,
#                         "themes": themes,
#                         "score": score,
#                         "popularity": popularity
#                     }})

#     for rec in data.get("Recommendations", []):
#         for rec_text in rec.get("Recommendation Texts", []):
#             rec_text = clean_text(rec_text)
#             if rec_text != "No content available":
#                 for chunk in chunk_text(rec_text):
#                     chunk_id = str(uuid.uuid4())
#                     chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                         "source": source_file,
#                         "item_id": item_id,
#                         "type": "recommendation",
#                         "genres": genres,
#                         "themes": themes,
#                         "score": score,
#                         "popularity": popularity
#                     }})

#     metadata.append({
#         "item_id": item_id,
#         "source": source_file,
#         "title": data.get("Title", "Unknown"),
#         "genres": genres,
#         "themes": themes,
#         "score": score,
#         "popularity": popularity,
#         "url": data.get("URL", "")
#     })
#     logger.info(f"Processed anime: {data.get('Title', 'Unknown')} with {len(chunks)} chunks")
#     return chunks, metadata

# def process_kdrama_kmovie(data, source_file):
#     """Process K-drama or K-movie JSON data."""
#     chunks = []
#     metadata = []
#     item_id = str(uuid.uuid4())

#     synopsis = clean_text(data.get("synopsis", ""))
#     genres = data.get("genres", [])
#     tags = data.get("tags", [])
#     title = data.get("title", "Unknown")
#     drama_info = data.get("drama_info", {})
#     rating = drama_info.get("rating", "")

#     if synopsis != "No content available":
#         for chunk in chunk_text(synopsis):
#             chunk_id = str(uuid.uuid4())
#             chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                 "source": source_file,
#                 "item_id": item_id,
#                 "type": "synopsis",
#                 "genres": genres,
#                 "tags": tags,
#                 "title": title,
#                 "rating": rating
#             }})

#     for review_type in ["completed", "dropped"]:
#         for review in data.get("reviews", {}).get(review_type, []):
#             if isinstance(review, str):
#                 review_text = clean_text(review)
#             else:
#                 review_text = clean_text(review.get("text", ""))
#             if review_text != "No content available":
#                 for chunk in chunk_text(review_text):
#                     chunk_id = str(uuid.uuid4())
#                     chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                         "source": source_file,
#                         "item_id": item_id,
#                         "type": f"review_{review_type}",
#                         "genres": genres,
#                         "tags": tags,
#                         "title": title,
#                         "rating": rating
#                     }})

#     for rec in data.get("recommendations", []):
#         rec_text = clean_text(rec.get("description", ""))
#         if rec_text != "No content available":
#             for chunk in chunk_text(rec_text):
#                 chunk_id = str(uuid.uuid4())
#                 chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                     "source": source_file,
#                     "item_id": item_id,
#                     "type": "recommendation",
#                     "genres": genres,
#                     "tags": tags,
#                     "title": title,
#                     "rating": rating
#                 }})

#     metadata.append({
#         "item_id": item_id,
#         "source": source_file,
#         "title": title,
#         "genres": genres,
#         "tags": tags,
#         "rating": rating,
#         "link": data.get("link", "")
#     })
#     logger.info(f"Processed kdrama/kmovie: {title} with {len(chunks)} chunks")
#     return chunks, metadata

# def process_manga(data, source_file):
#     """Process manga JSON data."""
#     chunks = []
#     metadata = []

#     for manga_item in data:
#         item_id = str(uuid.uuid4())

#         logger.info(f"Processing manga entry with URL: {manga_item.get('URL', 'Unknown')}")

#         description = clean_text(manga_item.get("Description", ""))
#         genres = manga_item.get("Genres", [])
#         statistics = manga_item.get("Statistics", {})
#         score = statistics.get("Score", "")
#         popularity = statistics.get("Popularity", "")

#         if description != "No content available":
#             for chunk in chunk_text(description):
#                 chunk_id = str(uuid.uuid4())
#                 chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                     "source": source_file,
#                     "item_id": item_id,
#                     "type": "description",
#                     "genres": genres,
#                     "score": score,
#                     "popularity": popularity
#                 }})

#         for review_type in ["Recommended", "Mixed Feelings", "Not Recommended"]:
#             for review in manga_item.get("Reviews", {}).get(review_type, []):
#                 review_text = clean_text(review.get("Review", ""))
#                 if review_text != "No content available":
#                     for chunk in chunk_text(review_text):
#                         chunk_id = str(uuid.uuid4())
#                         chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                             "source": source_file,
#                             "item_id": item_id,
#                             "type": f"review_{review_type.lower().replace(' ', '_')}",
#                             "genres": genres,
#                             "score": score,
#                             "popularity": popularity
#                         }})

#         for rec in manga_item.get("Recommendations", []):
#             for rec_text in rec.get("Recommendation Texts", []):
#                 rec_text = clean_text(rec_text)
#                 if rec_text != "No content available":
#                     for chunk in chunk_text(rec_text):
#                         chunk_id = str(uuid.uuid4())
#                         chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                             "source": source_file,
#                             "item_id": item_id,
#                             "type": "recommendation",
#                             "genres": genres,
#                             "score": score,
#                             "popularity": popularity
#                         }})

#         metadata.append({
#             "item_id": item_id,
#             "source": source_file,
#             "title": manga_item.get("Title", "Unknown"),
#             "genres": genres,
#             "score": score,
#             "popularity": popularity,
#             "url": manga_item.get("URL", "")
#         })
#         logger.info(f"Processed manga: {manga_item.get('Title', 'Unknown')} with {len(chunks)} chunks")
#     return chunks, metadata

# def process_hollywood_bollywood(data, source_file):
#     """Process Hollywood or Bollywood JSON data, including Trivia."""
#     chunks = []
#     metadata = []
#     item_id = str(uuid.uuid4())

#     plot = clean_text(data.get("Plot", ""))
#     genres = data.get("Genres", [])
#     title = data.get("Title", "Unknown")
#     rating = data.get("Rating", "")

#     if plot != "No content available":
#         for chunk in chunk_text(plot):
#             chunk_id = str(uuid.uuid4())
#             chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                 "source": source_file,
#                 "item_id": item_id,
#                 "type": "plot",
#                 "genres": genres,
#                 "rating": rating,
#                 "title": title
#             }})

#     for summary in data.get("PlotSummaries", []):
#         summary_text = clean_text(summary.get("Content", ""))
#         if summary_text != "No content available":
#             for chunk in chunk_text(summary_text):
#                 chunk_id = str(uuid.uuid4())
#                 chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                     "source": source_file,
#                     "item_id": item_id,
#                     "type": "plot_summary",
#                     "genres": genres,
#                     "rating": rating,
#                     "title": title
#                 }})

#     for review in data.get("Reviews", []):
#         review_text = clean_text(review.get("Content", ""))
#         review_rating = review.get("Rating", 0)
#         if review_text != "No content available":
#             for chunk in chunk_text(review_text):
#                 chunk_id = str(uuid.uuid4())
#                 chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                     "source": source_file,
#                     "item_id": item_id,
#                     "type": "review",
#                     "genres": genres,
#                     "rating": review_rating,
#                     "title": title
#                 }})

#     for trivia in data.get("Trivia", []):
#         trivia_text = clean_text(trivia)
#         if trivia_text != "No content available":
#             for chunk in chunk_text(trivia_text):
#                 chunk_id = str(uuid.uuid4())
#                 chunks.append({"id": chunk_id, "text": chunk, "metadata": {
#                     "source": source_file,
#                     "item_id": item_id,
#                     "type": "trivia",
#                     "genres": genres,
#                     "rating": rating,
#                     "title": title
#                 }})

#     metadata.append({
#         "item_id": item_id,
#         "source": source_file,
#         "title": title,
#         "genres": genres,
#         "rating": rating,
#         "url": data.get("URL", "")
#     })
#     logger.info(f"Processed hollywood/bollywood: {title} with {len(chunks)} chunks")
#     return chunks, metadata

# def main():
#     """Main function to process all JSON files."""
#     logger.info("Starting preprocessing...")
#     all_chunks = []
#     all_metadata = []

#     for file_name in os.listdir(RAW_DATA_DIR):
#         if file_name.endswith(".json"):
#             file_path = os.path.join(RAW_DATA_DIR, file_name)
#             logger.info(f"Processing file: {file_path}")
#             with open(file_path, "r", encoding="utf-8") as f:
#                 try:
#                     data = json.load(f)
#                 except json.JSONDecodeError as e:
#                     logger.error(f"Failed to parse {file_name}: {e}")
#                     continue

#             if file_name == "anime.json":
#                 if isinstance(data, list):
#                     for item in data:
#                         chunks, metadata = process_anime(item, file_name)
#                         all_chunks.extend(chunks)
#                         all_metadata.extend(metadata)
#                 else:
#                     chunks, metadata = process_anime(data, file_name)
#                     all_chunks.extend(chunks)
#                     all_metadata.extend(metadata)
#             elif file_name in ["kdrama.json", "kmovie.json"]:
#                 if isinstance(data, list):
#                     for item in data:
#                         chunks, metadata = process_kdrama_kmovie(item, file_name)
#                         all_chunks.extend(chunks)
#                         all_metadata.extend(metadata)
#                 else:
#                     chunks, metadata = process_kdrama_kmovie(data, file_name)
#                     all_chunks.extend(chunks)
#                     all_metadata.extend(metadata)
#             elif file_name == "manga.json":
#                 if isinstance(data, list):
#                     chunks, metadata = process_manga(data, file_name)
#                     all_chunks.extend(chunks)
#                     all_metadata.extend(metadata)
#                 else:
#                     logger.error(f"Unexpected format in manga.json: expected a list")
#             elif file_name in ["hollywood.json", "bollywood.json"]:
#                 if isinstance(data, list):
#                     for item in data:
#                         chunks, metadata = process_hollywood_bollywood(item, file_name)
#                         all_chunks.extend(chunks)
#                         all_metadata.extend(metadata)
#                 else:
#                     chunks, metadata = process_hollywood_bollywood(data, file_name)
#                     all_chunks.extend(chunks)
#                     all_metadata.extend(metadata)

#     with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
#         for chunk in all_chunks:
#             f.write(json.dumps(chunk) + "\n")
#     logger.info(f"Processed {len(all_chunks)} chunks and saved to {OUTPUT_CHUNKS_FILE}")

#     with open(METADATA_FILE, "w", encoding="utf-8") as f:
#         json.dump(all_metadata, f, indent=2)
#     logger.info(f"Saved {len(all_metadata)} metadata entries to {METADATA_FILE}")

# if __name__ == "__main__":
#     main()