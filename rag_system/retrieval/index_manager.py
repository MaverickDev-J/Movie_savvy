import os
import json
import numpy as np
import faiss
import logging
import asyncio
import yaml
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

class IndexManager:
    """Singleton class to manage FAISS index loading and caching"""
    _instance = None
    _index = None
    _metadata = None
    _chunks = None
    _model = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_config(self):
        """Load configuration from rag_config.yaml"""
        if self._config is None:
            try:
                CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "rag_config.yaml")
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load rag_config.yaml: {e}")
                raise e
        return self._config
    
    async def get_index_and_data(self):
        """Get cached index and data, load if not already cached"""
        if self._index is None or self._metadata is None or self._chunks is None:
            await self._load_index_once()
        return self._index, self._metadata, self._chunks
    
    async def get_model(self):
        """Get cached model, load if not already cached"""
        if self._model is None:
            config = self._load_config()
            model_name = config['retrieval']['embedding_model']
            logger.info(f"Loading SentenceTransformer model: {model_name} (first time)")
            self._model = await asyncio.to_thread(SentenceTransformer, model_name)
            logger.info("Model loaded and cached successfully")
        return self._model
    
    async def _load_index_once(self):
        """Load FAISS index, metadata, and chunks once"""
        logger.info("Loading FAISS index and metadata for the first time...")
        
        config = self._load_config()
        BASE_DIR = Path(__file__).resolve().parent.parent
        
        # Define paths from config
        INDEX_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['index_file'])
        INDEX_METADATA_FILE = str(BASE_DIR / config['retrieval']['index']['save_dir'] / config['retrieval']['index']['metadata_file'])
        CHUNKS_FILE = os.path.join(config['data']['processed_dir'], config['data']['preprocessing']['output_file'])
        
        try:
            # Load index
            self._index = await asyncio.to_thread(faiss.read_index, str(INDEX_FILE))
            
            # Load metadata
            with open(INDEX_METADATA_FILE, 'r') as f:
                self._metadata = json.load(f)
            
            # Load chunks
            with open(CHUNKS_FILE, 'r') as f:
                self._chunks = {json.loads(line)['id']: json.loads(line) for line in f}
            
            logger.info("Index and metadata loaded and cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to load index or metadata: {e}")
            raise e

# Global singleton instance
index_manager = IndexManager()