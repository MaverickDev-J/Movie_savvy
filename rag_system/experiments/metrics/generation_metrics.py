import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import logging
from typing import List, Dict, Any
import re
import numpy as np
from collections import Counter

# Try to import textstat, fallback if not available
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logging.warning("textstat not available, some fluency metrics will be limited")

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class GenerationMetrics:
    """Calculates comprehensive generation metrics for the RAG system."""
    
    def __init__(self):
        """Initialize with NLTK stopwords for relevance calculation."""
        self.stop_words = set(stopwords.words('english'))
        logger.info("GenerationMetrics initialized")
    
    def calculate_metrics(self, query: str, response: str, context: List[str], generation_time: float) -> Dict[str, float]:
        """Calculate comprehensive generation metrics for a single query.
        
        Args:
            query: The input query string.
            response: The generated response string.
            context: List of context strings used for generation.
            generation_time: Time taken for generation in seconds.
        
        Returns:
            Dictionary of calculated metrics.
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics['generation_time'] = generation_time
            
            if not response.strip():
                logger.warning("Empty response generated")
                metrics.update({
                    'response_length': 0,
                    'response_length_chars': 0,
                    'response_sentences': 0,
                    'response_coherence': 0.0,
                    'response_relevance': 0.0,
                    'query_relevance': 0.0,
                    'context_relevance': 0.0,
                    'repetition_score': 0.0,
                    'fluency_score': 0.0,
                    'lexical_diversity': 0.0,
                    'avg_sentence_length': 0.0,
                    'sentence_length_variance': 0.0
                })
                return metrics
            
            # Length metrics
            response_words = word_tokenize(response.lower())
            response_words_clean = [w for w in response_words if w.isalpha()]
            metrics['response_length'] = len(response_words_clean)
            metrics['response_length_chars'] = len(response)
            
            sentences = sent_tokenize(response)
            metrics['response_sentences'] = len(sentences)
            
            # Sentence length metrics
            if sentences:
                sent_lengths = [len(word_tokenize(sent)) for sent in sentences]
                metrics['avg_sentence_length'] = float(np.mean(sent_lengths))
                metrics['sentence_length_variance'] = float(np.var(sent_lengths))
                
                # Coherence based on sentence length consistency
                if len(sent_lengths) > 1:
                    length_cv = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0
                    metrics['response_coherence'] = max(0.0, 1.0 - length_cv)  # Lower CV = higher coherence
                else:
                    metrics['response_coherence'] = 1.0
            else:
                metrics['avg_sentence_length'] = 0.0
                metrics['sentence_length_variance'] = 0.0
                metrics['response_coherence'] = 0.0
            
            # Relevance metrics
            query_tokens = set(word_tokenize(query.lower())) - self.stop_words
            query_tokens = {w for w in query_tokens if w.isalpha()}
            
            context_text = ' '.join(context) if context else ''
            context_tokens = set(word_tokenize(context_text.lower())) - self.stop_words
            context_tokens = {w for w in context_tokens if w.isalpha()}
            
            response_tokens = set(response_words_clean) - self.stop_words
            
            # Query relevance (overlap between query and response)
            if query_tokens:
                query_overlap = len(query_tokens & response_tokens) / len(query_tokens)
                metrics['query_relevance'] = query_overlap
            else:
                metrics['query_relevance'] = 0.0
            
            # Context relevance (overlap between context and response)
            if context_tokens:
                context_overlap = len(context_tokens & response_tokens) / len(context_tokens)
                metrics['context_relevance'] = context_overlap
            else:
                metrics['context_relevance'] = 0.0
            
            # Combined relevance
            metrics['response_relevance'] = (metrics['query_relevance'] + metrics['context_relevance']) / 2
            
            # Repetition and diversity metrics
            total_words = len(response_words_clean)
            unique_words = len(set(response_words_clean))
            metrics['repetition_score'] = unique_words / total_words if total_words > 0 else 1.0
            
            # Lexical diversity (Type-Token Ratio)
            metrics['lexical_diversity'] = metrics['repetition_score']  # Same as repetition score
            
            # Advanced repetition detection
            metrics['bigram_repetition'] = self._calculate_ngram_repetition(response_words_clean, 2)
            metrics['trigram_repetition'] = self._calculate_ngram_repetition(response_words_clean, 3)
            
            # Fluency metrics
            if TEXTSTAT_AVAILABLE:
                try:
                    flesch_score = flesch_reading_ease(response)
                    metrics['fluency_score'] = max(0.0, min(1.0, flesch_score / 100.0))  # Normalize to [0,1]
                    
                    grade_level = flesch_kincaid_grade(response)
                    metrics['grade_level'] = grade_level
                except:
                    metrics['fluency_score'] = 0.5  # Default
                    metrics['grade_level'] = 10.0  # Default grade level
            else:
                # Simple fluency approximation
                metrics['fluency_score'] = self._simple_fluency_score(response)
                metrics['grade_level'] = 10.0  # Default
            
            # Content quality metrics
            metrics['information_density'] = self._calculate_information_density(response, query_tokens, context_tokens)
            metrics['specificity_score'] = self._calculate_specificity(response)
            
            # Answer completeness (based on question type)
            metrics['answer_completeness'] = self._calculate_answer_completeness(query, response)
            
            # Factual consistency indicator (simple heuristic)
            metrics['factual_consistency'] = self._calculate_factual_consistency(response, context)
            
            logger.debug(f"Generation metrics calculated: {list(metrics.keys())}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating generation metrics: {e}")
            return {'error_occurred': 1.0, 'generation_time': generation_time}
    
    def _calculate_ngram_repetition(self, words: List[str], n: int) -> float:
        """Calculate n-gram repetition rate.
        
        Args:
            words: List of words in the response.
            n: N-gram size.
            
        Returns:
            Repetition rate (0 = no repetition, 1 = complete repetition).
        """
        try:
            if len(words) < n:
                return 0.0
            
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            if not ngrams:
                return 0.0
            
            ngram_counts = Counter(ngrams)
            repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
            
            return repeated_ngrams / len(ngrams)
            
        except Exception:
            return 0.0
    
    def _simple_fluency_score(self, text: str) -> float:
        """Calculate a simple fluency score when textstat is not available.
        
        Args:
            text: The text to analyze.
            
        Returns:
            Fluency score between 0 and 1.
        """
        try:
            # Simple heuristics for fluency
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.0
            
            # Penalize very short or very long sentences
            sent_lengths = [len(word_tokenize(sent)) for sent in sentences]
            avg_sent_length = np.mean(sent_lengths)
            
            # Optimal sentence length is around 15-20 words
            length_score = 1.0 - abs(avg_sent_length - 17.5) / 17.5
            length_score = max(0.0, min(1.0, length_score))
            
            # Check for proper sentence structure (basic punctuation)
            punctuation_score = len(re.findall(r'[.!?]', text)) / len(sentences)
            punctuation_score = min(1.0, punctuation_score)
            
            return (length_score + punctuation_score) / 2
            
        except Exception:
            return 0.5
    
    def _calculate_information_density(self, response: str, query_tokens: set, context_tokens: set) -> float:
        """Calculate information density of the response.
        
        Args:
            response: The generated response.
            query_tokens: Set of query tokens.
            context_tokens: Set of context tokens.
            
        Returns:
            Information density score.
        """
        try:
            response_words = word_tokenize(response.lower())
            response_words = [w for w in response_words if w.isalpha() and w not in self.stop_words]
            
            if not response_words:
                return 0.0
            
            # Count informative words (those that appear in query or context)
            informative_words = sum(1 for word in response_words if word in query_tokens or word in context_tokens)
            
            return informative_words / len(response_words)
            
        except Exception:
            return 0.0
    
    def _calculate_specificity(self, response: str) -> float:
        """Calculate specificity of the response (presence of specific details).
        
        Args:
            response: The generated response.
            
        Returns:
            Specificity score.
        """
        try:
            # Simple heuristics for specificity
            specific_patterns = [
                r'\b\d{4}\b',  # Years
                r'\b\d+\.\d+\b',  # Decimal numbers
                r'\b\d+%\b',  # Percentages
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper nouns (names)
                r'"\w+[^"]*"',  # Quoted text
            ]
            
            specific_count = 0
            for pattern in specific_patterns:
                specific_count += len(re.findall(pattern, response))
            
            words = word_tokenize(response)
            if not words:
                return 0.0
            
            # Normalize by response length
            return min(1.0, specific_count / (len(words) / 10))  # Per 10 words
            
        except Exception:
            return 0.0
    
    def _calculate_answer_completeness(self, query: str, response: str) -> float:
        """Calculate how complete the answer is based on the query type.
        
        Args:
            query: The input query.
            response: The generated response.
            
        Returns:
            Completeness score.
        """
        try:
            query_lower = query.lower()
            response_lower = response.lower()
            
            # Check for question words and their typical answer patterns
            completeness_indicators = {
                'what': ['is', 'are', 'was', 'were', 'means', 'refers'],
                'who': ['is', 'was', 'are', 'were'],
                'when': ['in', 'on', 'at', 'during', 'since', 'until'],
                'where': ['in', 'at', 'on', 'near', 'located'],
                'why': ['because', 'due to', 'since', 'as', 'reason'],
                'how': ['by', 'through', 'using', 'via', 'method', 'way']
            }
            
            score = 0.5  # Base score
            
            for question_word, indicators in completeness_indicators.items():
                if question_word in query_lower:
                    # Check if response contains appropriate answer indicators
                    indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
                    if indicator_count > 0:
                        score += 0.3
                    break
            
            # Additional completeness checks
            if len(response.split()) >= 10:  # Reasonable length
                score += 0.1
            
            if any(punct in response for punct in '.!?'):  # Proper punctuation
                score += 0.1
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _calculate_factual_consistency(self, response: str, context: List[str]) -> float:
        """Calculate a simple factual consistency score.
        
        Args:
            response: The generated response.
            context: List of context strings.
            
        Returns:
            Factual consistency score.
        """
        try:
            if not context:
                return 0.5  # Default when no context available
            
            response_words = set(word_tokenize(response.lower()))
            response_words = {w for w in response_words if w.isalpha() and w not in self.stop_words}
            
            context_text = ' '.join(context)
            context_words = set(word_tokenize(context_text.lower()))
            context_words = {w for w in context_words if w.isalpha() and w not in self.stop_words}
            
            if not response_words:
                return 0.0
            
            # Calculate overlap between response and context
            overlap = len(response_words & context_words)
            consistency_score = overlap / len(response_words)
            
            return min(1.0, consistency_score)
            
        except Exception:
            return 0.5
    
    def calculate_batch_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregated metrics across multiple generation results.
        
        Args:
            results: List of individual generation results with metrics.
            
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