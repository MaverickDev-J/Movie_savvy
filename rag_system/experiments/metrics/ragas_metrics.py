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