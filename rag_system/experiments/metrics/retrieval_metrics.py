import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
from typing import List, Dict, Any
from collections import Counter
import time
import re

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """Calculates comprehensive retrieval metrics for the RAG system."""
    
    def __init__(self):
        """Initialize with NLTK stopwords for diversity calculation."""
        self.stop_words = set(stopwords.words('english'))
        logger.info("RetrievalMetrics initialized")
    
    def calculate_metrics(self, query: str, retrieved_chunks: List[Dict[str, Any]], retrieval_time: float) -> Dict[str, float]:
        """Calculate comprehensive retrieval metrics for a single query.
        
        Args:
            query: The input query string.
            retrieved_chunks: List of retrieved chunks with 'text', 'similarity', and 'source'.
            retrieval_time: Time taken for retrieval in seconds.
        
        Returns:
            Dictionary of calculated metrics.
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics['retrieval_time'] = retrieval_time
            metrics['chunks_retrieved_count'] = len(retrieved_chunks)
            
            if not retrieved_chunks:
                logger.warning("No chunks retrieved for query")
                metrics.update({
                    'avg_similarity_score': 0.0,
                    'similarity_variance': 0.0,
                    'similarity_std': 0.0,
                    'min_similarity': 0.0,
                    'max_similarity': 0.0,
                    'context_diversity': 0.0,
                    'context_coverage': 0.0,
                    'avg_chunk_length': 0.0,
                    'total_context_length': 0.0,
                    'unique_sources_count': 0
                })
                return metrics
            
            # Similarity metrics
            similarity_scores = [chunk.get('similarity', 0.0) for chunk in retrieved_chunks]
            metrics['avg_similarity_score'] = float(np.mean(similarity_scores))
            metrics['similarity_variance'] = float(np.var(similarity_scores))
            metrics['similarity_std'] = float(np.std(similarity_scores))
            metrics['min_similarity'] = float(np.min(similarity_scores))
            metrics['max_similarity'] = float(np.max(similarity_scores))
            
            # Context diversity metrics
            all_text = ' '.join(chunk.get('text', '') for chunk in retrieved_chunks)
            query_tokens = set(word_tokenize(query.lower()))
            query_tokens = {word for word in query_tokens if word.isalpha() and word not in self.stop_words}
            
            # Tokenize all retrieved text
            all_tokens = word_tokenize(all_text.lower())
            content_tokens = [word for word in all_tokens if word.isalpha() and word not in self.stop_words]
            unique_terms = set(content_tokens)
            
            # Context diversity (unique terms / total terms)
            metrics['context_diversity'] = len(unique_terms) / len(content_tokens) if content_tokens else 0.0
            
            # Context coverage (how many query terms are covered by retrieved context)
            if query_tokens:
                covered_terms = query_tokens.intersection(unique_terms)
                metrics['context_coverage'] = len(covered_terms) / len(query_tokens)
            else:
                metrics['context_coverage'] = 0.0
            
            # Length metrics
            chunk_lengths = [len(chunk.get('text', '')) for chunk in retrieved_chunks]
            metrics['avg_chunk_length'] = float(np.mean(chunk_lengths))
            metrics['total_context_length'] = sum(chunk_lengths)
            
            # Source distribution metrics
            sources = [chunk.get('source', 'unknown') for chunk in retrieved_chunks]
            unique_sources = set(sources)
            metrics['unique_sources_count'] = len(unique_sources)
            
            # Source entropy (measure of source diversity)
            source_counts = Counter(sources)
            total_chunks = len(retrieved_chunks)
            source_probs = [count / total_chunks for count in source_counts.values()]
            metrics['source_entropy'] = -sum(p * np.log2(p) for p in source_probs if p > 0)
            
            # Relevance distribution (how well distributed are the similarity scores)
            if len(similarity_scores) > 1:
                # Coefficient of variation for similarity scores
                metrics['similarity_cv'] = metrics['similarity_std'] / metrics['avg_similarity_score'] if metrics['avg_similarity_score'] > 0 else 0.0
            else:
                metrics['similarity_cv'] = 0.0
            
            # Quality metrics
            # Percentage of chunks above average similarity
            avg_sim = metrics['avg_similarity_score']
            above_avg_count = sum(1 for score in similarity_scores if score >= avg_sim)
            metrics['above_avg_similarity_ratio'] = above_avg_count / len(similarity_scores)
            
            # Content redundancy (approximate measure using n-gram overlap)
            metrics['content_redundancy'] = self._calculate_content_redundancy(retrieved_chunks)
            
            logger.debug(f"Retrieval metrics calculated: {list(metrics.keys())}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating retrieval metrics: {e}")
            return {'error_occurred': 1.0, 'retrieval_time': retrieval_time}
    
    def _calculate_content_redundancy(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Calculate content redundancy using simple text similarity.
        
        Args:
            retrieved_chunks: List of retrieved chunks.
            
        Returns:
            Redundancy score (0 = no redundancy, 1 = complete redundancy).
        """
        try:
            if len(retrieved_chunks) < 2:
                return 0.0
            
            texts = [chunk.get('text', '') for chunk in retrieved_chunks]
            similarities = []
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    # Simple Jaccard similarity on word level
                    text1_words = set(word_tokenize(texts[i].lower()))
                    text2_words = set(word_tokenize(texts[j].lower()))
                    
                    intersection = len(text1_words & text2_words)
                    union = len(text1_words | text2_words)
                    
                    similarity = intersection / union if union > 0 else 0.0
                    similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating content redundancy: {e}")
            return 0.0
    
    def calculate_batch_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregated metrics across multiple retrieval results.
        
        Args:
            results: List of individual retrieval results with metrics.
            
        Returns:
            Dictionary of aggregated metrics.
        """
        try:
            if not results:
                return {}
            
            # Aggregate individual metrics
            aggregated = {}
            metric_keys = set()
            for result in results:
                metric_keys.update(result.keys())
            
            for key in metric_keys:
                values = [result.get(key, 0) for result in results if isinstance(result.get(key), (int, float))]
                if values:
                    aggregated[f"avg_{key}"] = float(np.mean(values))
                    aggregated[f"std_{key}"] = float(np.std(values))
                    aggregated[f"min_{key}"] = float(np.min(values))
                    aggregated[f"max_{key}"] = float(np.max(values))
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error calculating batch metrics: {e}")
            return {}