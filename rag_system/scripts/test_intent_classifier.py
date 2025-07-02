import logging
from pathlib import Path
from rag_system.functions.intent_classifier import EnhancedIntentClassifier

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent / "rag_system"
log_dir = BASE_DIR / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "test_intent.log"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_intent_classifier():
    """Test EnhancedIntentClassifier with sample queries."""
    classifier = EnhancedIntentClassifier()
    test_queries = [
        "Dune 2024 trailer",
        "Compare Avengers and Justice League",
        "What is the plot of Attack on Titan",
        "Behind the scenes of Pushpa 2"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        result = classifier.analyze_query_intent(query)
        logger.info(f"Result: {result}")

if __name__ == "__main__":
    try:
        test_intent_classifier()
    except Exception as e:
        logger.error(f"Test failed: {e}")