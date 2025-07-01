# import time
# import logging
# from typing import Dict, List, Any, Optional
# from collections import Counter
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# logger = logging.getLogger(__name__)

# class RetrievalMetrics:
#     """Calculates metrics for document retrieval quality and performance."""
    
#     def __init__(self):
#         """Initialize the RetrievalMetrics class."""
#         try:
#             nltk.data.find('tokenizers/punkt')
#             nltk.data.find('corpora/stopwords')
#         except LookupError:
#             nltk.download('punkt')
#             nltk.download('stopwords')
        
#         self.stop_words = set(stopwords.words('english'))
#         logger.info("RetrievalMetrics initialized")
    
#     def calculate_metrics(self, 
#                          query: str, 
#                          retrieved_chunks: List[Dict[str, Any]], 
#                          retrieval_time: float) -> Dict[str, float]:
#         """Calculate comprehensive retrieval metrics.
        
#         Args:
#             query: The user's query
#             retrieved_chunks: List of retrieved document chunks with metadata
#             retrieval_time: Time taken for retrieval in seconds
            
#         Returns:
#             Dictionary containing all calculated metrics
#         """
#         if not retrieved_chunks:
#             logger.warning("No chunks retrieved")
#             return {
#                 'retrieval_time': retrieval_time,
#                 'chunks_retrieved_count': 0,
#                 'avg_similarity_score': 0.0,
#                 'similarity_variance': 0.0,
#                 'context_diversity': 0.0,
#                 'source_distribution': {}
#             }
        
#         metrics = {}
        
#         # Basic retrieval metrics
#         metrics['retrieval_time'] = retrieval_time
#         metrics['chunks_retrieved_count'] = len(retrieved_chunks)
        
#         # Similarity metrics
#         similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in retrieved_chunks]
#         metrics['avg_similarity_score'] = np.mean(similarity_scores) if similarity_scores else 0.0
#         metrics['max_similarity_score'] = np.max(similarity_scores) if similarity_scores else 0.0
#         metrics['min_similarity_score'] = np.min(similarity_scores) if similarity_scores else 0.0
#         metrics['similarity_variance'] = np.var(similarity_scores) if len(similarity_scores) > 1 else 0.0
#         metrics['similarity_std'] = np.std(similarity_scores) if len(similarity_scores) > 1 else 0.0
        
#         # Context diversity metrics
#         metrics['context_diversity'] = self._calculate_context_diversity(retrieved_chunks)
#         metrics['content_overlap'] = self._calculate_content_overlap(retrieved_chunks)
        
#         # Source distribution metrics
#         source_distribution = self._calculate_source_distribution(retrieved_chunks)
#         metrics['source_distribution'] = source_distribution
#         metrics['source_diversity'] = len(source_distribution)
#         metrics['source_entropy'] = self._calculate_source_entropy(source_distribution)
        
#         # Query-chunk relevance metrics
#         metrics['query_chunk_relevance'] = self._calculate_query_chunk_relevance(query, retrieved_chunks)
#         metrics['chunk_quality_score'] = self._calculate_chunk_quality(retrieved_chunks)
        
#         # Ranking quality metrics
#         metrics['ranking_quality'] = self._calculate_ranking_quality(similarity_scores)
#         metrics['top_k_precision'] = self._calculate_top_k_precision(similarity_scores, k=3)
        
#         # Coverage metrics
#         metrics['query_coverage'] = self._calculate_query_coverage(query, retrieved_chunks)
#         metrics['topic_coverage'] = self._calculate_topic_coverage(retrieved_chunks)
        
#         return metrics
    
#     def _calculate_context_diversity(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate diversity of retrieved contexts."""
#         try:
#             if len(retrieved_chunks) <= 1:
#                 return 0.0
            
#             texts = [chunk.get('text', '') or chunk.get('content', '') for chunk in retrieved_chunks]
#             texts = [t for t in texts if t]
            
#             if not texts:
#                 return 0.0
            
#             # Calculate pairwise similarities
#             similarities = []
#             for i in range(len(texts)):
#                 for j in range(i + 1, len(texts)):
#                     similarity = self._calculate_text_similarity(texts[i], texts[j])
#                     similarities.append(similarity)
            
#             if not similarities:
#                 return 0.0
            
#             # Diversity is inverse of average similarity
#             avg_similarity = np.mean(similarities)
#             diversity = 1.0 - avg_similarity
            
#             return max(0.0, min(diversity, 1.0))
            
#         except Exception as e:
#             logger.error(f"Error calculating context diversity: {e}")
#             return 0.0
    
#     def _calculate_text_similarity(self, text1: str, text2: str) -> float:
#         """Calculate similarity between two texts using word overlap."""
#         try:
#             words1 = set(word_tokenize(text1.lower())) - self.stop_words
#             words2 = set(word_tokenize(text2.lower())) - self.stop_words
            
#             if not words1 or not words2:
#                 return 0.0
            
#             intersection = len(words1 & words2)
#             union = len(words1 | words2)
            
#             return intersection / union if union > 0 else 0.0
#         except:
#             return 0.0
    
#     def _calculate_content_overlap(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate content overlap between retrieved chunks."""
#         try:
#             if len(retrieved_chunks) <= 1:
#                 return 0.0
            
#             texts = [chunk.get('text', '') or chunk.get('content', '') for chunk in retrieved_chunks]
#             texts = [t for t in texts if t]
            
#             if not texts:
#                 return 0.0
            
#             # Calculate word-level overlap
#             word_sets = []
#             for text in texts:
#                 words = set(word_tokenize(text.lower())) - self.stop_words
#                 word_sets.append(words)
            
#             if not word_sets:
#                 return 0.0
            
#             # Calculate average pairwise overlap
#             overlaps = []
#             for i in range(len(word_sets)):
#                 for j in range(i + 1, len(word_sets)):
#                     if word_sets[i] and word_sets[j]:
#                         intersection = len(word_sets[i] & word_sets[j])
#                         union = len(word_sets[i] | word_sets[j])
#                         overlap = intersection / union if union > 0 else 0.0
#                         overlaps.append(overlap)
            
#             return np.mean(overlaps) if overlaps else 0.0
            
#         except Exception as e:
#             logger.error(f"Error calculating content overlap: {e}")
#             return 0.0
    
#     def _calculate_source_distribution(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
#         """Calculate distribution of sources in retrieved chunks."""
#         try:
#             sources = []
#             for chunk in retrieved_chunks:
#                 source = chunk.get('source', 'unknown')
#                 if isinstance(source, dict):
#                     source = source.get('name', 'unknown')
#                 sources.append(str(source))
            
#             if not sources:
#                 return {}
            
#             source_counts = Counter(sources)
#             total_chunks = len(sources)
            
#             source_distribution = {
#                 source: count / total_chunks 
#                 for source, count in source_counts.items()
#             }
            
#             return source_distribution
            
#         except Exception as e:
#             logger.error(f"Error calculating source distribution: {e}")
#             return {}
    
#     def _calculate_source_entropy(self, source_distribution: Dict[str, float]) -> float:
#         """Calculate entropy of source distribution."""
#         try:
#             if not source_distribution:
#                 return 0.0
            
#             proportions = list(source_distribution.values())
#             if not proportions:
#                 return 0.0
            
#             # Calculate Shannon entropy
#             entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
            
#             # Normalize by maximum possible entropy
#             max_entropy = np.log2(len(proportions)) if len(proportions) > 1 else 1.0
#             normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
#             return normalized_entropy
            
#         except Exception as e:
#             logger.error(f"Error calculating source entropy: {e}")
#             return 0.0
    
#     def _calculate_query_chunk_relevance(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate average relevance of chunks to the query."""
#         try:
#             if not retrieved_chunks:
#                 return 0.0
            
#             query_words = set(word_tokenize(query.lower())) - self.stop_words
            
#             relevance_scores = []
#             for chunk in retrieved_chunks:
#                 text = chunk.get('text', '') or chunk.get('content', '')
#                 if text:
#                     chunk_words = set(word_tokenize(text.lower())) - self.stop_words
                    
#                     if query_words and chunk_words:
#                         intersection = len(query_words & chunk_words)
#                         union = len(query_words | chunk_words)
#                         relevance = intersection / union if union > 0 else 0.0
#                         relevance_scores.append(relevance)
            
#             return np.mean(relevance_scores) if relevance_scores else 0.0
            
#         except Exception as e:
#             logger.error(f"Error calculating query-chunk relevance: {e}")
#             return 0.0
    
#     def _calculate_chunk_quality(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate overall quality of retrieved chunks."""
#         try:
#             if not retrieved_chunks:
#                 return 0.0
            
#             quality_scores = []
#             for chunk in retrieved_chunks:
#                 text = chunk.get('text', '') or chunk.get('content', '')
#                 if not text:
#                     quality_scores.append(0.0)
#                     continue
                
#                 # Simple quality metrics
#                 words = text.split()
#                 quality = 0.0
                
#                 # Length appropriateness (50-500 words is good)
#                 word_count = len(words)
#                 if 50 <= word_count <= 500:
#                     quality += 0.5
#                 elif 20 <= word_count <= 1000:
#                     quality += 0.3
                
#                 # Content word ratio
#                 content_words = [w for w in word_tokenize(text.lower()) 
#                                if w not in self.stop_words and w.isalpha()]
#                 if words:
#                     content_ratio = len(content_words) / len(words)
#                     quality += content_ratio * 0.5
                
#                 quality_scores.append(min(quality, 1.0))
            
#             return np.mean(quality_scores) if quality_scores else 0.0
            
#         except Exception as e:
#             logger.error(f"Error calculating chunk quality: {e}")
#             return 0.0
    
#     def _calculate_ranking_quality(self, similarity_scores: List[float]) -> float:
#         """Calculate quality of ranking based on similarity scores."""
#         try:
#             if len(similarity_scores) <= 1:
#                 return 1.0
            
#             # Check if scores are in descending order
#             correctly_ordered = sum(1 for i in range(len(similarity_scores) - 1) 
#                                   if similarity_scores[i] >= similarity_scores[i + 1])
            
#             total_pairs = len(similarity_scores) - 1
#             ranking_quality = correctly_ordered / total_pairs if total_pairs > 0 else 1.0
            
#             return ranking_quality
            
#         except Exception as e:
#             logger.error(f"Error calculating ranking quality: {e}")
#             return 0.0
    
#     def _calculate_top_k_precision(self, similarity_scores: List[float], k: int = 3) -> float:
#         """Calculate precision at k based on similarity scores."""
#         try:
#             if not similarity_scores or k <= 0:
#                 return 0.0
            
#             top_k_scores = similarity_scores[:min(k, len(similarity_scores))]
#             threshold = 0.5
#             relevant_in_top_k = sum(1 for score in top_k_scores if score > threshold)
            
#             precision_at_k = relevant_in_top_k / len(top_k_scores)
#             return precision_at_k
            
#         except Exception as e:
#             logger.error(f"Error calculating top-k precision: {e}")
#             return 0.0
    
#     def _calculate_query_coverage(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate how well retrieved chunks cover query terms."""
#         try:
#             if not retrieved_chunks:
#                 return 0.0
            
#             query_words = set(word_tokenize(query.lower())) - self.stop_words
#             if not query_words:
#                 return 0.0
            
#             # Collect all words from retrieved chunks
#             all_chunk_words = set()
#             for chunk in retrieved_chunks:
#                 text = chunk.get('text', '') or chunk.get('content', '')
#                 if text:
#                     chunk_words = set(word_tokenize(text.lower())) - self.stop_words
#                     all_chunk_words.update(chunk_words)
            
#             # Calculate coverage
#             covered_words = query_words & all_chunk_words
#             coverage = len(covered_words) / len(query_words) if query_words else 0.0
            
#             return coverage
            
#         except Exception as e:
#             logger.error(f"Error calculating query coverage: {e}")
#             return 0.0
    
#     def _calculate_topic_coverage(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate topic coverage diversity in retrieved chunks."""
#         try:
#             if not retrieved_chunks:
#                 return 0.0
            
#             # Simple topic coverage based on text diversity
#             texts = [chunk.get('text', '') or chunk.get('content', '') for chunk in retrieved_chunks]
#             texts = [t for t in texts if t]
            
#             if not texts:
#                 return 0.0
            
#             # Use text diversity as proxy for topic coverage
#             return self._calculate_context_diversity(retrieved_chunks)
            
#         except Exception as e:
#             logger.error(f"Error calculating topic coverage: {e}")
#             return 0.0











# import logging
# from typing import Dict, List, Any
# import numpy as np
# from collections import Counter
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# logger = logging.getLogger(__name__)

# class RetrievalMetrics:
#     """Calculates metrics for document retrieval quality and performance in a RAG system."""
    
#     def __init__(self):
#         """Initialize the RetrievalMetrics class with NLTK resources."""
#         try:
#             nltk.data.find('tokenizers/punkt')
#             nltk.data.find('corpora/stopwords')
#         except LookupError:
#             nltk.download('punkt')
#             nltk.download('stopwords')
        
#         self.stop_words = set(stopwords.words('english'))
#         logger.info("RetrievalMetrics initialized")
    
#     def calculate_metrics(self, 
#                          query: str, 
#                          retrieved_chunks: List[Dict[str, Any]], 
#                          retrieval_time: float) -> Dict[str, float]:
#         """Calculate retrieval metrics.
        
#         Args:
#             query: The user's query string
#             retrieved_chunks: List of retrieved chunks with metadata (e.g., 'text', 'similarity', 'source')
#             retrieval_time: Time taken for retrieval in seconds
            
#         Returns:
#             Dictionary of calculated metrics
#         """
#         if not retrieved_chunks:
#             logger.warning("No chunks retrieved")
#             return {
#                 'retrieval_time': retrieval_time,
#                 'chunks_retrieved_count': 0,
#                 'avg_similarity_score': 0.0,
#                 'similarity_variance': 0.0,
#                 'context_diversity': 0.0,
#                 'source_distribution': {},
#                 'query_coverage': 0.0
#             }
        
#         metrics = {
#             'retrieval_time': retrieval_time,
#             'chunks_retrieved_count': len(retrieved_chunks)
#         }
        
#         # Similarity metrics
#         similarity_scores = [chunk.get('similarity', chunk.get('similarity_score', 0.0)) 
#                            for chunk in retrieved_chunks]
#         metrics['avg_similarity_score'] = np.mean(similarity_scores) if similarity_scores else 0.0
#         metrics['max_similarity_score'] = np.max(similarity_scores) if similarity_scores else 0.0
#         metrics['min_similarity_score'] = np.min(similarity_scores) if similarity_scores else 0.0
#         metrics['similarity_variance'] = np.var(similarity_scores) if len(similarity_scores) > 1 else 0.0
        
#         # Diversity metric
#         metrics['context_diversity'] = self._calculate_context_diversity(retrieved_chunks)
        
#         # Source distribution
#         source_distribution = self._calculate_source_distribution(retrieved_chunks)
#         metrics['source_distribution'] = source_distribution
#         metrics['source_diversity'] = len(source_distribution)
        
#         # Query coverage
#         metrics['query_coverage'] = self._calculate_query_coverage(query, retrieved_chunks)
        
#         return metrics
    
#     def _calculate_context_diversity(self, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate diversity based on unique sources."""
#         try:
#             sources = [chunk.get('source', 'unknown') for chunk in retrieved_chunks]
#             unique_sources = len(set(sources))
#             return unique_sources / len(retrieved_chunks) if retrieved_chunks else 0.0
#         except Exception as e:
#             logger.error(f"Error calculating context diversity: {e}")
#             return 0.0
    
#     def _calculate_source_distribution(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
#         """Calculate distribution of sources in retrieved chunks."""
#         try:
#             sources = []
#             for chunk in retrieved_chunks:
#                 source = chunk.get('source', 'unknown')
#                 if isinstance(source, dict):
#                     source = source.get('name', 'unknown')
#                 sources.append(str(source))
            
#             source_counts = Counter(sources)
#             total_chunks = len(retrieved_chunks)
#             return {source: count / total_chunks for source, count in source_counts.items()}
#         except Exception as e:
#             logger.error(f"Error calculating source distribution: {e}")
#             return {}
    
#     def _calculate_query_coverage(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
#         """Calculate how well retrieved chunks cover query terms."""
#         try:
#             query_words = set(word_tokenize(query.lower())) - self.stop_words
#             if not query_words:
#                 return 0.0
            
#             covered_words = set()
#             for chunk in retrieved_chunks:
#                 text = chunk.get('text', '') or chunk.get('content', '')
#                 chunk_words = set(word_tokenize(text.lower())) - self.stop_words
#                 covered_words.update(chunk_words & query_words)
            
#             return len(covered_words) / len(query_words)
#         except Exception as e:
#             logger.error(f"Error calculating query coverage: {e}")
#             return 0.0





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