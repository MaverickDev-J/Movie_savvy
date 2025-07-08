from rag_system.embeddings.chunk_embeddings import ChunkEmbeddingManager
from rag_system.functions.youtube_handler import YouTubeHandler
import logging

logger = logging.getLogger(__name__)


def populate_index():
    youtube_handler = YouTubeHandler()
    embedding_manager = ChunkEmbeddingManager()
    videos = youtube_handler.search_videos("dune 2024 trailer")
    chunks = []
    for video in videos:
        transcript = youtube_handler.get_transcript(video['id'])
        chunks.extend(youtube_handler.chunk_transcript(transcript))
    embeddings = embedding_manager.embed_chunks(chunks)
    embedding_manager.index.train(embeddings)  # Train IVF index
    embedding_manager.add_to_index(chunks, embeddings)
    embedding_manager.save_index()

if __name__ == "__main__":
    populate_index()