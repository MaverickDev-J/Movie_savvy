# """Utility to build the final prompt with memory, vector context, Reddit context, and query."""

# from typing import List, Optional
# from pathlib import Path
# import logging
# import yaml

# # Configuration paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"

# # Load configuration
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# # Logging setup
# log_dir = BASE_DIR / config["pipeline"]["logging"]["log_dir"]
# log_dir.mkdir(parents=True, exist_ok=True)
# log_file = log_dir / config["pipeline"]["logging"]["log_file"]
# logging.basicConfig(
#     level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
# )
# logger = logging.getLogger(__name__)

# def build_prompt(
#     memory_context: str,
#     vector_context: List[str],
#     reddit_context: Optional[List[str]],
#     current_query: str,
#     use_reddit: bool = False
# ) -> str:
#     """
#     Build the final prompt using memory, vector context, Reddit context, and query.

#     Args:
#         memory_context (str): Formatted conversation history from ConversationMemory.
#         vector_context (List[str]): List of retrieved document chunks from vector store.
#         reddit_context (Optional[List[str]]): List of Reddit content items, if applicable.
#         current_query (str): The current user query.
#         use_reddit (bool): Whether to use the Reddit-specific prompt template.

#     Returns:
#         str: Formatted prompt ready for the generator.
#     """
#     # Load prompt templates
#     prompt_dir = BASE_DIR / config["generation"]["prompt"]["template_dir"]
#     default_prompt_file = config["generation"]["prompt"]["default_prompt_file"]
#     reddit_prompt_file = config["generation"]["prompt"]["reddit_prompt_file"]

#     # Read the appropriate template
#     template_path = prompt_dir / (reddit_prompt_file if use_reddit and reddit_context else default_prompt_file)
#     try:
#         with open(template_path, "r") as f:
#             template = f.read()
#     except Exception as e:
#         logger.error(f"Failed to load prompt template {template_path}: {e}")
#         # Fallback to default template
#         template = (
#             "You are a movie-savvy AI assistant specializing in entertainment content (movies, anime, manga, etc.). "
#             "Answer the user's query strictly based on the provided context, avoiding any invented details. "
#             "If the context is insufficient, state so clearly and provide a concise response based on available information.\n\n"
#             "Conversation history:\n{memory}\n\n"
#             "Context Information:\n{context}\n\n"
#             "User Question: {query}\n\n"
#             "Instructions:\n"
#             "- Use only the provided context for specific details.\n"
#             "- Be concise, informative, and maintain a fun, movie-geek tone.\n"
#             "- For recommendations or comparisons, explain relevance briefly.\n"
#             "- If context lacks details, say 'Insufficient context for a detailed response' and provide a general answer.\n\n"
#             "### Response:"
#         ) if not use_reddit else (
#             "You are a movie-savvy AI specializing in films, anime, manga, and TV shows, with a fun, geeky tone. "
#             "Answer the query using the provided context sources, avoiding any invented details. "
#             "If context is insufficient, clearly state so and provide a concise, accurate response based on general entertainment knowledge.\n\n"
#             "Conversation history:\n{memory}\n\n"
#             "Official Data:\n{vector_context}\n\n"
#             "Community Discussion:\n{reddit_context}\n\n"
#             "Query: {query}\n\n"
#             "Instructions:\n"
#             "- Use Official Data for facts, ratings, plot details, and technical information\n"
#             "- Use Community Discussion for opinions, reactions, and fan perspectives\n"
#             "- Prioritize context for specific details; do not fabricate information\n"
#             "- Deliver concise, engaging answers in a movie-enthusiast tone\n"
#             "- For recommendations or comparisons, explain relevance briefly\n"
#             "- If context lacks details, say 'Limited context available' and give a general answer\n"
#             "- Clearly distinguish between factual information and community opinions\n\n"
#             "### Response:"
#         )

#     # Format context
#     vector_text = "\n".join(vector_context) if vector_context else "No official data available."
#     reddit_text = "\n".join(reddit_context) if reddit_context and use_reddit else "No community discussion available."
#     memory_text = memory_context.strip() or "(no prior conversation)"

#     # Prepare template variables
#     context_dict = {
#         "memory": memory_text,
#         "query": current_query.strip(),
#         "context": vector_text,  # Used in default template
#         "vector_context": vector_text,  # Used in Reddit template
#         "reddit_context": reddit_text  # Used in Reddit template
#     }

#     try:
#         prompt = template.format(**context_dict)
#         logger.debug(f"Built prompt:\n{prompt}")
#         return prompt
#     except KeyError as e:
#         logger.error(f"Template formatting failed due to missing key: {e}")
#         return template  # Return unformatted template as fallback





"""Utility to build the final prompt with memory, vector context, Reddit context, and query."""

from typing import List, Optional
from pathlib import Path
import logging
import yaml

# Configuration paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "rag_config.yaml"

# Load configuration
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Logging setup
log_dir = BASE_DIR / config["pipeline"]["logging"]["log_dir"]
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / config["pipeline"]["logging"]["log_file"]
logging.basicConfig(
    level=logging.getLevelName(config["pipeline"]["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def build_prompt(
    memory_context: str,
    vector_context: List[str],
    reddit_context: Optional[List[str]],
    current_query: str,
    use_reddit: bool = False
) -> str:
    """
    Build the final prompt using memory, vector context, Reddit context, and query.

    Args:
        memory_context (str): Formatted conversation history from ConversationMemory.
        vector_context (List[str]): List of retrieved document chunks from vector store.
        reddit_context (Optional[List[str]]): List of Reddit content items, if applicable.
        current_query (str): The current user query.
        use_reddit (bool): Whether to use the Reddit-specific prompt template.

    Returns:
        str: Formatted prompt ready for the generator.
    """
    # Load prompt templates
    prompt_dir = BASE_DIR / config["generation"]["prompt"]["template_dir"]
    default_prompt_file = config["generation"]["prompt"]["default_prompt_file"]
    reddit_prompt_file = config["generation"]["prompt"]["reddit_prompt_file"]

    # Read the appropriate template
    template_path = prompt_dir / (reddit_prompt_file if use_reddit and reddit_context else default_prompt_file)
    try:
        with open(template_path, "r") as f:
            template = f.read()
    except Exception as e:
        logger.error(f"Failed to load prompt template {template_path}: {e}")
        # Fallback to default template
        template = (
            """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to provide concise, engaging, and accurate answers in a conversational, geeky tone. Use only the provided context to answer the query. If the context is insufficient, say so clearly and provide a brief, factual response based on general entertainment knowledge.

Conversation History: {memory}

Instruction: {query}

Context: {context}

Response Guidelines:
- Answer directly using the context, avoiding invented details.
- Keep responses short (2–4 sentences) unless a detailed explanation is requested.
- Use a fun, movie-enthusiast tone, like chatting with a fellow fan.
- If context is missing or irrelevant, state: "Not enough context for details, but here's what I can share."
- Format as a natural narrative, not a list, unless the query asks for one.

### Response:"""
        ) if not use_reddit else (
            """You are a movie-savvy AI specializing in movies, anime, manga, and TV shows, trained to deliver concise, engaging answers in a geeky, conversational tone. Answer the query using official data for facts and Reddit discussions for fan opinions, without inventing details. If information is limited, say so clearly and provide a brief, factual response.

Conversation History: {memory}

Instruction: {query}

Official Data: {vector_context}

Community Discussion: {reddit_context}

Response Guidelines:
- Use official data for plot, cast, ratings, or technical details.
- Use Reddit discussion for fan reactions or opinions, clearly labeling them as "fans say" or "Reddit users mention."
- Keep responses concise (2–4 sentences) unless a detailed breakdown is requested.
- If no relevant data is available, state: "Limited info available, but here's a general take."
- Maintain a fun, movie-enthusiast tone, blending facts and fan insights naturally.
- Avoid lists unless the query explicitly asks for them.

### Response:"""
        )

    # Format context
    vector_text = "\n".join(vector_context) if vector_context else "No official data available."
    reddit_text = "\n".join(reddit_context) if reddit_context and use_reddit else "No community discussion available."
    memory_text = memory_context.strip() or "(no prior conversation)"

    # Prepare template variables
    context_dict = {
        "memory": memory_text,
        "query": current_query.strip(),
        "context": vector_text,  # Used in default template
        "vector_context": vector_text,  # Used in Reddit template
        "reddit_context": reddit_text  # Used in Reddit template
    }

    try:
        prompt = template.format(**context_dict)
        logger.debug(f"Built prompt:\n{prompt}")
        return prompt
    except KeyError as e:
        logger.error(f"Template formatting failed due to missing key: {e}")
        return template  # Return unformatted template as fallback