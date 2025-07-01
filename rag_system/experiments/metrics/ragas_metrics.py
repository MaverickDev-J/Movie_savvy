# import logging
# import re
# from typing import Dict, List, Any, Optional
# import numpy as np
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# import nltk

# logger = logging.getLogger(__name__)

# class RAGASMetrics:
#     """Implements RAGAS (RAG Assessment) metrics for evaluating RAG systems.
    
#     RAGAS metrics include:
#     - Faithfulness: How faithful is the response to the context
#     - Answer Relevancy: How relevant is the response to the query
#     - Context Precision: How precise is the retrieved context
#     - Context Recall: How much relevant context was retrieved
#     - Context Entity Recall: Entity-level recall in context
#     """
    
#     def __init__(self):
#         """Initialize the RAGASMetrics class."""
#         try:
#             nltk.data.find('tokenizers/punkt')
#             nltk.data.find('corpora/stopwords')
#         except LookupError:
#             nltk.download('punkt')
#             nltk.download('stopwords')
        
#         self.stop_words = set(stopwords.words('english'))
#         logger.info("RAGASMetrics initialized")
    
#     def calculate_metrics(self,
#                          query: str,
#                          response: str,
#                          context: List[str],
#                          ground_truth: Optional[str] = None) -> Dict[str, float]:
#         """Calculate all RAGAS metrics.
        
#         Args:
#             query: The user's query
#             response: The generated response
#             context: List of context chunks used for generation
#             ground_truth: Optional ground truth answer for comparison
            
#         Returns:
#             Dictionary containing all RAGAS metrics
#         """
#         metrics = {}
        
#         try:
#             # Core RAGAS metrics
#             metrics['faithfulness'] = self.calculate_faithfulness(response, context)
#             metrics['answer_relevancy'] = self.calculate_answer_relevancy(query, response)
#             metrics['context_precision'] = self.calculate_context_precision(query, context)
#             metrics['context_recall'] = self.calculate_context_recall(query, context, ground_truth)
#             metrics['context_entity_recall'] = self.calculate_context_entity_recall(query, context)
            
#             # Additional derived metrics
#             metrics['overall_ragas_score'] = self._calculate_overall_score(metrics)
            
#         except Exception as e:
#             logger.error(f"Error calculating RAGAS metrics: {e}")
#             # Return default values on error
#             metrics = {
#                 'faithfulness': 0.0,
#                 'answer_relevancy': 0.0,
#                 'context_precision': 0.0,
#                 'context_recall': 0.0,
#                 'context_entity_recall': 0.0,
#                 'overall_ragas_score': 0.0
#             }
        
#         return metrics
    
#     def calculate_faithfulness(self, response: str, context: List[str]) -> float:
#         """Calculate faithfulness score.
        
#         Measures how faithful the response is to the provided context.
#         A faithful response should not contain information that contradicts
#         or is not supported by the context.
        
#         Args:
#             response: The generated response
#             context: List of context chunks
            
#         Returns:
#             Faithfulness score between 0 and 1
#         """
#         try:
#             if not response or not context:
#                 return 0.0
            
#             # Combine all context
#             combined_context = " ".join(context)
            
#             # Split response into claims/sentences
#             response_sentences = sent_tokenize(response)
#             if not response_sentences:
#                 return 0.0
            
#             faithful_count = 0
            
#             for sentence in response_sentences:
#                 # Check if sentence is supported by context
#                 if self._is_sentence_supported(sentence, combined_context):
#                     faithful_count += 1
            
#             faithfulness = faithful_count / len(response_sentences)
#             return min(faithfulness, 1.0)
            
#         except Exception as e:
#             logger.error(f"Error calculating faithfulness: {e}")
#             return 0.0
    
#     def calculate_answer_relevancy(self, query: str, response: str) -> float:
#         """Calculate answer relevancy score.
        
#         Measures how relevant the response is to the query.
        
#         Args:
#             query: The user's query
#             response: The generated response
            
#         Returns:
#             Answer relevancy score between 0 and 1
#         """
#         try:
#             if not query or not response:
#                 return 0.0
            
#             # Extract keywords from query and response
#             query_words = set(word_tokenize(query.lower())) - self.stop_words
#             response_words = set(word_tokenize(response.lower())) - self.stop_words
            
#             if not query_words:
#                 return 0.0
            
#             # Calculate word overlap
#             common_words = query_words & response_words
#             word_overlap = len(common_words) / len(query_words)
            
#             # Calculate semantic similarity (simplified)
#             semantic_score = self._calculate_semantic_similarity(query, response)
            
#             # Combine scores
#             relevancy = (word_overlap * 0.4) + (semantic_score * 0.6)
            
#             return min(relevancy, 1.0)
            
#         except Exception as e:
#             logger.error(f"Error calculating answer relevancy: {e}")
#             return 0.0
    
#     def calculate_context_precision(self, query: str, context: List[str]) -> float:
#         """Calculate context precision score.
        
#         Measures how precise the retrieved context is for the query.
#         Higher precision means less irrelevant context.
        
#         Args:
#             query: The user's query
#             context: List of context chunks
            
#         Returns:
#             Context precision score between 0 and 1
#         """
#         try:
#             if not query or not context:
#                 return 0.0
            
#             query_words = set(word_tokenize(query.lower())) - self.stop_words
            
#             if not query_words:
#                 return 0.0
            
#             relevant_chunks = 0
            
#             for chunk in context:
#                 chunk_words = set(word_tokenize(chunk.lower())) - self.stop_words
                
#                 # Check relevance based on word overlap
#                 if chunk_words:
#                     overlap = len(query_words & chunk_words) / len(query_words)
#                     if overlap > 0.1:  # Threshold for relevance
#                         relevant_chunks += 1
            
#             precision = relevant_chunks / len(context) if context else 0.0
#             return min(precision, 1.0)
            
#         except Exception as e:
#             logger.error(f"Error calculating context precision: {e}")
#             return 0.0
    
#     def calculate_context_recall(self, query: str, context: List[str], ground_truth: Optional[str] = None) -> float:
#         """Calculate context recall score.
        
#         Measures how much of the relevant information was retrieved.
#         Without ground truth, we estimate based on query coverage.
        
#         Args:
#             query: The user's query
#             context: List of context chunks
#             ground_truth: Optional ground truth answer
            
#         Returns:
#             Context recall score between 0 and 1
#         """
#         try:
#             if not query or not context:
#                 return 0.0
            
#             if ground_truth:
#                 return self._calculate_recall_with_ground_truth(context, ground_truth)
#             else:
#                 return self._calculate_recall_without_ground_truth(query, context)
            
#         except Exception as e:
#             logger.error(f"Error calculating context recall: {e}")
#             return 0.0
    
#     def calculate_context_entity_recall(self, query: str, context: List[str]) -> float:
#         """Calculate context entity recall score.
        
#         Measures how well the context covers entities mentioned in the query.
        
#         Args:
#             query: The user's query
#             context: List of context chunks
            
#         Returns:
#             Context entity recall score between 0 and 1
#         """
#         try:
#             if not query or not context:
#                 return 0.0
            
#             # Extract entities from query (simplified)
#             query_entities = self._extract_entities(query)
            
#             if not query_entities:
#                 return 1.0  # No entities to recall
            
#             # Check which entities are covered in context
#             combined_context = " ".join(context).lower()
#             covered_entities = 0
            
#             for entity in query_entities:
#                 if entity.lower() in combined_context:
#                     covered_entities += 1
            
#             entity_recall = covered_entities / len(query_entities)
#             return min(entity_recall, 1.0)
            
#         except Exception as e:
#             logger.error(f"Error calculating context entity recall: {e}")
#             return 0.0
    
#     def _is_sentence_supported(self, sentence: str, context: str) -> bool:
#         """Check if a sentence is supported by the context."""
#         # Simplified faithfulness check
#         sentence_words = set(word_tokenize(sentence.lower())) - self.stop_words
#         context_words = set(word_tokenize(context.lower())) - self.stop_words
        
#         if not sentence_words:
#             return True
        
#         # Check word overlap
#         overlap = len(sentence_words & context_words) / len(sentence_words)
        
#         # Also check for key phrases
#         key_phrases = self._extract_key_phrases(sentence)
#         phrase_support = 0
        
#         for phrase in key_phrases:
#             if phrase.lower() in context.lower():
#                 phrase_support += 1
        
#         phrase_score = phrase_support / len(key_phrases) if key_phrases else 0
        
#         # Combine word overlap and phrase support
#         support_score = (overlap * 0.6) + (phrase_score * 0.4)
        
#         return support_score > 0.3  # Threshold for support
    
#     def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
#         """Calculate semantic similarity between two texts (simplified)."""
#         # Simple word overlap based similarity
#         words1 = set(word_tokenize(text1.lower())) - self.stop_words
#         words2 = set(word_tokenize(text2.lower())) - self.stop_words
        
#         if not words1 or not words2:
#             return 0.0
        
#         intersection = len(words1 & words2)
#         union = len(words1 | words2)
        
#         return intersection / union if union > 0 else 0.0
    
#     def _calculate_recall_with_ground_truth(self, context: List[str], ground_truth: str) -> float:
#         """Calculate recall when ground truth is available."""
#         combined_context = " ".join(context)
        
#         # Extract key information from ground truth
#         gt_words = set(word_tokenize(ground_truth.lower())) - self.stop_words
#         context_words = set(word_tokenize(combined_context.lower())) - self.stop_words
        
#         if not gt_words:
#             return 1.0
        
#         covered_words = len(gt_words & context_words)
#         recall = covered_words / len(gt_words)
        
#         return min(recall, 1.0)
    
#     def _calculate_recall_without_ground_truth(self, query: str, context: List[str]) -> float:
#         """Estimate recall without ground truth based on query coverage."""
#         # This is an approximation - check how well context addresses query aspects
#         query_words = set(word_tokenize(query.lower())) - self.stop_words
#         combined_context = " ".join(context)
#         context_words = set(word_tokenize(combined_context.lower())) - self.stop_words
        
#         if not query_words:
#             return 1.0
        
#         covered_query_words = len(query_words & context_words)
#         coverage = covered_query_words / len(query_words)
        
#         # Boost score if context seems comprehensive
#         context_length_bonus = min(len(combined_context.split()) / 200, 0.2)
        
#         estimated_recall = coverage + context_length_bonus
#         return min(estimated_recall, 1.0)
    
#     def _extract_entities(self, text: str) -> List[str]:
#         """Extract entities from text (simplified approach)."""
#         # Simple entity extraction - look for capitalized words and common patterns
#         entities = []
        
#         # Capitalized words (potential named entities)
#         words = word_tokenize(text)
#         for word in words:
#             if word[0].isupper() and len(word) > 1 and word.isalpha():
#                 entities.append(word)
        
#         # Numbers and dates
#         numbers = re.findall(r'\b\d+\b', text)
#         entities.extend(numbers)
        
#         # Common patterns
#         patterns = [
#             r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Person names
#             r'\b\d{4}\b',  # Years
#         ]
        
#         for pattern in patterns:
#             matches = re.findall(pattern, text)
#             entities.extend(matches)
        
#         return list(set(entities))  # Remove duplicates
    
#     def _extract_key_phrases(self, text: str) -> List[str]:
#         """Extract key phrases from text."""
#         # Simple key phrase extraction
#         sentences = sent_tokenize(text)
#         phrases = []
        
#         for sentence in sentences:
#             words = word_tokenize(sentence)
#             # Extract noun phrases (simplified)
#             for i in range(len(words) - 1):
#                 if words[i].isalpha() and words[i+1].isalpha():
#                     phrase = f"{words[i]} {words[i+1]}"
#                     phrases.append(phrase)
        
#         return phrases
    
#     def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
#         """Calculate overall RAGAS score."""
#         core_metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        
#         scores = []
#         for metric in core_metrics:
#             if metric in metrics and isinstance(metrics[metric], (int, float)):
#                 scores.append(metrics[metric])
        
#         if not scores:
#             return 0.0
        
#         # Weighted average (faithfulness and relevancy are more important)
#         weights = [0.3, 0.3, 0.2, 0.2]  # faithfulness, relevancy, precision, recall
        
#         if len(scores) == len(weights):
#             overall_score = sum(score * weight for score, weight in zip(scores, weights))
#         else:
#             overall_score = np.mean(scores)
        
#         return min(overall_score, 1.0)
















# import logging
# import re
# from typing import Dict, List, Optional
# import numpy as np
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# import nltk

# logger = logging.getLogger(__name__)

# class RAGASMetrics:
#     """Implements RAGAS (RAG Assessment) metrics for evaluating RAG systems.
    
#     RAGAS metrics include:
#     - Faithfulness: How faithful the response is to the context
#     - Answer Relevancy: How relevant the response is to the query
#     - Context Precision: How precise the retrieved context is
#     - Context Recall: How much relevant context was retrieved
#     - Context Entity Recall: Entity-level recall in context
#     - Overall RAGAS Score: Weighted average of core metrics
#     """
    
#     def __init__(self):
#         """Initialize the RAGASMetrics class with NLTK resources."""
#         try:
#             nltk.data.find('tokenizers/punkt')
#             nltk.data.find('corpora/stopwords')
#         except LookupError:
#             nltk.download('punkt')
#             nltk.download('stopwords')
        
#         self.stop_words = set(stopwords.words('english'))
#         logger.info("RAGASMetrics initialized")
    
#     def calculate_metrics(self,
#                          query: str,
#                          response: str,
#                          context: List[str],
#                          ground_truth: Optional[str] = None) -> Dict[str, float]:
#         """Calculate all RAGAS metrics.
        
#         Args:
#             query: The user's query
#             response: The generated response
#             context: List of context chunks used for generation
#             ground_truth: Optional ground truth answer for comparison
            
#         Returns:
#             Dictionary containing all RAGAS metrics
#         """
#         metrics = {
#             'faithfulness': 0.0,
#             'answer_relevancy': 0.0,
#             'context_precision': 0.0,
#             'context_recall': 0.0,
#             'context_entity_recall': 0.0,
#             'overall_ragas_score': 0.0
#         }
        
#         try:
#             metrics['faithfulness'] = self.calculate_faithfulness(response, context)
#             metrics['answer_relevancy'] = self.calculate_answer_relevancy(query, response)
#             metrics['context_precision'] = self.calculate_context_precision(query, context)
#             metrics['context_recall'] = self.calculate_context_recall(query, context, ground_truth)
#             metrics['context_entity_recall'] = self.calculate_context_entity_recall(query, context)
#             metrics['overall_ragas_score'] = self._calculate_overall_score(metrics)
            
#         except Exception as e:
#             logger.error(f"Error calculating RAGAS metrics: {e}")
        
#         return metrics
    
#     def calculate_faithfulness(self, response: str, context: List[str]) -> float:
#         """Calculate faithfulness score based on context support.
        
#         Args:
#             response: The generated response
#             context: List of context chunks
            
#         Returns:
#             Faithfulness score between 0 and 1
#         """
#         if not response or not context:
#             return 0.0
        
#         combined_context = " ".join(context).lower()
#         response_sentences = sent_tokenize(response)
        
#         if not response_sentences:
#             return 0.0
        
#         faithful_count = 0
#         for sentence in response_sentences:
#             sentence_words = set(word_tokenize(sentence.lower())) - self.stop_words
#             context_words = set(word_tokenize(combined_context)) - self.stop_words
#             overlap = len(sentence_words & context_words) / len(sentence_words) if sentence_words else 0.0
#             if overlap > 0.3:  # Threshold for support
#                 faithful_count += 1
        
#         return faithful_count / len(response_sentences)
    
#     def calculate_answer_relevancy(self, query: str, response: str) -> float:
#         """Calculate answer relevancy based on query-response overlap.
        
#         Args:
#             query: The user's query
#             response: The generated response
            
#         Returns:
#             Answer relevancy score between 0 and 1
#         """
#         if not query or not response:
#             return 0.0
        
#         query_words = set(word_tokenize(query.lower())) - self.stop_words
#         response_words = set(word_tokenize(response.lower())) - self.stop_words
        
#         if not query_words or not response_words:
#             return 0.0
        
#         overlap = len(query_words & response_words) / len(query_words)
#         return min(overlap, 1.0)
    
#     def calculate_context_precision(self, query: str, context: List[str]) -> float:
#         """Calculate context precision based on relevant chunks.
        
#         Args:
#             query: The user's query
#             context: List of context chunks
            
#         Returns:
#             Context precision score between 0 and 1
#         """
#         if not query or not context:
#             return 0.0
        
#         query_words = set(word_tokenize(query.lower())) - self.stop_words
#         if not query_words:
#             return 0.0
        
#         relevant_chunks = 0
#         for chunk in context:
#             chunk_words = set(word_tokenize(chunk.lower())) - self.stop_words
#             if chunk_words and len(query_words & chunk_words) / len(query_words) > 0.1:
#                 relevant_chunks += 1
        
#         return relevant_chunks / len(context) if context else 0.0
    
#     def calculate_context_recall(self, query: str, context: List[str], ground_truth: Optional[str] = None) -> float:
#         """Calculate context recall with or without ground truth.
        
#         Args:
#             query: The user's query
#             context: List of context chunks
#             ground_truth: Optional ground truth answer
            
#         Returns:
#             Context recall score between 0 and 1
#         """
#         if not query or not context:
#             return 0.0
        
#         combined_context = " ".join(context).lower()
#         if ground_truth:
#             gt_words = set(word_tokenize(ground_truth.lower())) - self.stop_words
#             context_words = set(word_tokenize(combined_context)) - self.stop_words
#             return len(gt_words & context_words) / len(gt_words) if gt_words else 1.0
        
#         query_words = set(word_tokenize(query.lower())) - self.stop_words
#         context_words = set(word_tokenize(combined_context)) - self.stop_words
#         return len(query_words & context_words) / len(query_words) if query_words else 1.0
    
#     def calculate_context_entity_recall(self, query: str, context: List[str]) -> float:
#         """Calculate entity recall based on query entities in context.
        
#         Args:
#             query: The user's query
#             context: List of context chunks
            
#         Returns:
#             Context entity recall score between 0 and 1
#         """
#         if not query or not context:
#             return 0.0
        
#         query_entities = self._extract_entities(query)
#         if not query_entities:
#             return 1.0
        
#         combined_context = " ".join(context).lower()
#         covered_entities = sum(1 for entity in query_entities if entity.lower() in combined_context)
#         return covered_entities / len(query_entities)
    
#     def _extract_entities(self, text: str) -> List[str]:
#         """Extract simple entities from text (capitalized words, numbers)."""
#         words = word_tokenize(text)
#         entities = [word for word in words if (word[0].isupper() and word.isalpha()) or word.isdigit()]
#         return list(set(entities))
    
#     def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
#         """Calculate overall RAGAS score as a weighted average.
        
#         Args:
#             metrics: Dictionary of calculated metrics
            
#         Returns:
#             Overall score between 0 and 1
#         """
#         core_metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
#         weights = [0.3, 0.3, 0.2, 0.2]  # Emphasize faithfulness and relevancy
#         scores = [metrics.get(metric, 0.0) for metric in core_metrics]
#         return min(sum(w * s for w, s in zip(weights, scores)), 1.0)














import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import re
import logging
from typing import Dict, List, Optional, Any, Set
import numpy as np
from collections import Counter
import string

# Optional semantic similarity support
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    util = None

# Ensure NLTK resources
nltk_downloads = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
for resource in nltk_downloads:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif resource == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif resource == 'averaged_perceptron_tagger':
            nltk.data.find('taggers/averaged_perceptron_tagger')
        elif resource == 'maxent_ne_chunker':
            nltk.data.find('chunkers/maxent_ne_chunker')
        elif resource == 'words':
            nltk.data.find('corpora/words')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Could not download {resource}: {e}")

logger = logging.getLogger(__name__)

class RagasMetrics:
    """Calculates RAGAS-inspired metrics for the RAG system."""
    
    def __init__(self, use_semantic_similarity: bool = True):
        """Initialize with stopwords and optional SentenceTransformer model.
        
        Args:
            use_semantic_similarity: Whether to use semantic similarity (requires sentence-transformers)
        """
        self.stop_words = set(stopwords.words('english'))
        self.use_semantic_similarity = use_semantic_similarity and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_semantic_similarity:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded for RAGAS metrics")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}. Falling back to keyword-based metrics.")
                self.model = None
                self.use_semantic_similarity = False
        else:
            self.model = None
            logger.info("Using keyword-based RAGAS metrics")
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and punctuation."""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _get_tokens(self, text: str, remove_stopwords: bool = True) -> Set[str]:
        """Extract tokens from text."""
        if not text:
            return set()
        
        tokens = word_tokenize(text.lower())
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words and token.isalpha()]
        
        return set(tokens)
    
    def _extract_named_entities(self, text: str) -> Set[str]:
        """Extract named entities from text using NLTK."""
        if not text:
            return set()
        
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags, binary=False)
            
            entities = set()
            for chunk in chunks:
                if isinstance(chunk, Tree):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    entities.add(entity.lower())
                else:
                    # Single word entities (proper nouns)
                    token, pos = chunk
                    if pos in ('NNP', 'NNPS'):
                        entities.add(token.lower())
            
            return entities
        except Exception as e:
            logger.warning(f"Error extracting named entities: {e}")
            return set()
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.use_semantic_similarity or not self.model:
            return self._keyword_similarity(text1, text2)
        
        try:
            emb1 = self.model.encode(text1, convert_to_tensor=True)
            emb2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return self._keyword_similarity(text1, text2)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate keyword-based similarity between two texts."""
        tokens1 = self._get_tokens(text1)
        tokens2 = self._get_tokens(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_faithfulness(self, response: str, context: List[str]) -> float:
        """Calculate faithfulness: fraction of response statements supported by context."""
        if not response or not context:
            return 0.0
        
        response_sentences = sent_tokenize(self._clean_text(response))
        if not response_sentences:
            return 0.0
        
        context_text = ' '.join(context)
        supported_count = 0
        
        for sentence in response_sentences:
            if not sentence.strip():
                continue
            
            # Check if sentence is supported by context
            similarity = self._semantic_similarity(sentence, context_text)
            
            # Threshold for considering a sentence as supported
            threshold = 0.6 if self.use_semantic_similarity else 0.3
            if similarity > threshold:
                supported_count += 1
        
        return supported_count / len(response_sentences)
    
    def _calculate_answer_relevancy(self, query: str, response: str) -> float:
        """Calculate answer relevancy: how relevant is the response to the query."""
        if not query or not response:
            return 0.0
        
        return self._semantic_similarity(query, response)
    
    def _calculate_context_precision(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """Calculate context precision: fraction of retrieved contexts that are relevant."""
        if not retrieved_chunks:
            return 0.0
        
        relevant_count = 0
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('text', '')
            if not chunk_text:
                continue
            
            # Use similarity score if available, otherwise calculate relevance
            if 'similarity' in chunk and chunk['similarity'] is not None:
                similarity = chunk['similarity']
            else:
                similarity = self._semantic_similarity(query, chunk_text)
            
            # Threshold for relevance
            if similarity > 0.5:
                relevant_count += 1
        
        return relevant_count / len(retrieved_chunks)
    
    def _calculate_context_recall(self, query: str, context: List[str], ground_truth: Optional[str] = None) -> float:
        """Calculate context recall: how much of the ground truth is covered by context."""
        if not ground_truth or not context:
            return 0.0
        
        context_text = ' '.join(context)
        return self._semantic_similarity(ground_truth, context_text)
    
    def _calculate_context_entity_recall(self, context: List[str], response: str) -> float:
        """Calculate context entity recall: fraction of context entities mentioned in response."""
        if not context or not response:
            return 0.0
        
        # Extract entities from context and response
        context_text = ' '.join(context)
        context_entities = self._extract_named_entities(context_text)
        response_entities = self._extract_named_entities(response)
        
        if not context_entities:
            return 1.0  # Perfect recall if no entities to recall
        
        # Calculate recall
        mentioned_entities = context_entities & response_entities
        return len(mentioned_entities) / len(context_entities)
    
    def _calculate_context_relevance(self, query: str, context: List[str]) -> float:
        """Calculate average relevance of context chunks to the query."""
        if not context:
            return 0.0
        
        relevance_scores = []
        for chunk in context:
            if chunk.strip():
                relevance = self._semantic_similarity(query, chunk)
                relevance_scores.append(relevance)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def _calculate_response_completeness(self, query: str, response: str, context: List[str]) -> float:
        """Calculate response completeness: how comprehensively the response addresses the query."""
        if not query or not response:
            return 0.0
        
        # Extract key concepts from query
        query_tokens = self._get_tokens(query, remove_stopwords=True)
        response_tokens = self._get_tokens(response, remove_stopwords=True)
        
        if not query_tokens:
            return 1.0
        
        # Calculate coverage of query concepts in response
        covered_concepts = query_tokens & response_tokens
        completeness = len(covered_concepts) / len(query_tokens)
        
        return completeness
    
    def _calculate_hallucination_score(self, response: str, context: List[str]) -> float:
        """Calculate hallucination score: amount of information in response not supported by context."""
        if not response or not context:
            return 0.0
        
        context_text = ' '.join(context)
        response_sentences = sent_tokenize(response)
        
        if not response_sentences:
            return 0.0
        
        unsupported_count = 0
        
        for sentence in response_sentences:
            if not sentence.strip():
                continue
            
            # Check if sentence has support in context
            similarity = self._semantic_similarity(sentence, context_text)
            
            # Low similarity indicates potential hallucination
            if similarity < 0.4:  # Threshold for support
                unsupported_count += 1
        
        return unsupported_count / len(response_sentences)
    
    def calculate_metrics(self, 
                         query: str, 
                         response: str, 
                         context: List[str], 
                         retrieved_chunks: List[Dict[str, Any]], 
                         ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Calculate comprehensive RAGAS metrics for a single query.
        
        Args:
            query: The input query string.
            response: The generated response string.
            context: List of context strings used for generation.
            retrieved_chunks: List of retrieved chunks with metadata.
            ground_truth: Optional ground truth response for comparison.
        
        Returns:
            Dictionary of calculated RAGAS metrics.
        """
        try:
            metrics = {}
            
            # Core RAGAS metrics
            metrics['faithfulness'] = self._calculate_faithfulness(response, context)
            metrics['answer_relevancy'] = self._calculate_answer_relevancy(query, response)
            metrics['context_precision'] = self._calculate_context_precision(query, retrieved_chunks)
            metrics['context_recall'] = self._calculate_context_recall(query, context, ground_truth)
            metrics['context_entity_recall'] = self._calculate_context_entity_recall(context, response)
            
            # Additional metrics
            metrics['context_relevance'] = self._calculate_context_relevance(query, context)
            metrics['response_completeness'] = self._calculate_response_completeness(query, response, context)
            metrics['hallucination_score'] = self._calculate_hallucination_score(response, context)
            
            # Composite scores
            # Overall quality score (higher is better)
            quality_components = [
                metrics['faithfulness'],
                metrics['answer_relevancy'],
                metrics['context_precision'],
                metrics['response_completeness']
            ]
            metrics['overall_quality'] = np.mean([score for score in quality_components if score is not None])
            
            # RAG pipeline effectiveness (considering both retrieval and generation)
            retrieval_effectiveness = (metrics['context_precision'] + metrics['context_relevance']) / 2
            generation_effectiveness = (metrics['faithfulness'] + metrics['answer_relevancy']) / 2
            metrics['rag_effectiveness'] = (retrieval_effectiveness + generation_effectiveness) / 2
            
            logger.debug(f"RAGAS metrics calculated: {list(metrics.keys())}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating RAGAS metrics: {e}")
            return {'error_occurred': 1.0}
    
    def get_metric_explanations(self) -> Dict[str, str]:
        """Get explanations for each metric."""
        return {
            'faithfulness': 'Fraction of response statements supported by context (0-1, higher is better)',
            'answer_relevancy': 'Semantic similarity between query and response (0-1, higher is better)',
            'context_precision': 'Fraction of retrieved contexts that are relevant to query (0-1, higher is better)',
            'context_recall': 'How much of ground truth is covered by retrieved context (0-1, higher is better)',
            'context_entity_recall': 'Fraction of context entities mentioned in response (0-1, higher is better)',
            'context_relevance': 'Average relevance of context chunks to query (0-1, higher is better)',
            'response_completeness': 'How comprehensively response addresses query (0-1, higher is better)',
            'hallucination_score': 'Amount of unsupported information in response (0-1, lower is better)',
            'overall_quality': 'Composite quality score (0-1, higher is better)',
            'rag_effectiveness': 'Overall RAG pipeline effectiveness (0-1, higher is better)'
        }