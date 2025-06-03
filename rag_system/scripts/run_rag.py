# import sys
# import json
# import logging
# from pathlib import Path

# # Add the parent directory to Python path so we can import our modules
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(BASE_DIR))

# from retrieval.retriever import retrieve
# from generation.generator import Generator
# import argparse

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('output/logs/rag_pipeline.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# OUTPUT_DIR = BASE_DIR / "output" / "results"

# def run_rag(query, top_k=5, max_new_tokens=256):
#     logger.info(f"Processing query: {query}")
#     try:
#         # Retrieve relevant chunks
#         chunks = retrieve(query, top_k)
#         logger.info(f"Retrieved {len(chunks)} chunks")

#         # Generate response
#         generator = Generator()
#         response = generator.generate(query, chunks, max_new_tokens=max_new_tokens)
        
#         # Save result
#         result = {
#             "query": query,
#             "response": response,
#             "retrieved_chunks": chunks
#         }
#         OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#         output_file = OUTPUT_DIR / f"result_{query[:20].replace(' ', '_')}.json"
#         with open(output_file, 'w') as f:
#             json.dump(result, f, indent=2)
#         logger.info(f"Saved result to {output_file}")

#         return response
#     except Exception as e:
#         logger.error(f"RAG pipeline failed: {e}")
#         return f"Error processing query: {e}"

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--query', default="Find anime similar to Berserk with dark themes")
#     parser.add_argument('--top_k', type=int, default=5)
#     parser.add_argument('--max_new_tokens', type=int, default=256)
#     args = parser.parse_args()
#     response = run_rag(args.query, args.top_k, args.max_new_tokens)
#     logger.info(f"Response: {response}")

# if __name__ == "__main__":
#     main()



import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from retrieval.retriever import retrieve
from generation.generator import Generator
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'output' / 'logs' / 'rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = BASE_DIR / "output" / "results"

def run_rag(query, top_k=5, max_new_tokens=512):
    logger.info(f"Processing query: {query}")
    try:
        chunks = retrieve(query, top_k)
        logger.info(f"Retrieved {len(chunks)} chunks")
        generator = Generator()
        response = generator.generate(query, chunks, max_new_tokens=max_new_tokens)
        result = {
            "query": query,
            "response": response,
            "retrieved_chunks": chunks
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"result_{query[:20].replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result to {output_file}")
        return response
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        return f"Error processing query: {e}"

def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline for entertainment queries")
    parser.add_argument('--query', default="Find anime similar to Berserk with dark themes", help="Query for the RAG system")
    parser.add_argument('--top_k', type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Maximum new tokens for generation")
    args = parser.parse_args()
    response = run_rag(args.query, args.top_k, args.max_new_tokens)
    logger.info(f"Response: {response}")

if __name__ == "__main__":
    main()