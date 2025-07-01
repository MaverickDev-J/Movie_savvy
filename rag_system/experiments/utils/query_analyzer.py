import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Analyzes user queries for the rag_system by extracting keywords and classifying query types.
    
    Attributes:
        stop_words: Set of English stop words to filter out from queries.
    """
    
    def __init__(self):
        """Initialize the QueryAnalyzer and ensure necessary NLTK resources are available."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze the query to extract keywords and determine its type.
        
        Args:
            query: The user's query string.
        
        Returns:
            A dictionary containing:
                - 'keywords': List of extracted keywords.
                - 'type': Classified query type (e.g., 'factual', 'opinion', 'current', 'recommendation').
        """
        if not query.strip():
            logger.warning("Empty query received")
            return {'keywords': [], 'type': 'unknown'}
        
        tokens = word_tokenize(query.lower())
        keywords = [word for word in tokens if word not in self.stop_words and word.isalpha()]
        
        if not keywords:
            logger.warning("Query contains only stop words")
            return {'keywords': [], 'type': 'unknown'}
        
# Enhanced classification with priority handling
        query_types = []
        if any(word in keywords for word in ['latest', 'recent', 'current', 'new']):
            query_types.append('current')
        if any(word in keywords for word in ['best', 'recommend', 'suggest', 'similar']):
            query_types.append('recommendation')
        if any(word in keywords for word in ['opinion', 'think', 'feel', 'compare']):
            query_types.append('opinion')

        # Prioritize current events, then recommendations, then opinions
        if 'current' in query_types:
            query_type = 'current'
        elif 'recommendation' in query_types:
            query_type = 'recommendation'
        elif 'opinion' in query_types:
            query_type = 'opinion'
        else:
            query_type = 'factual'
        
        return {'keywords': keywords, 'type': query_type}

# Example usage:
# analyzer = QueryAnalyzer()
# result = analyzer.analyze("What are the best anime movies of 2023?")
# print(result)  # Output: {'keywords': ['best', 'anime', 'movies', '2023'], 'type': 'recommendation'}