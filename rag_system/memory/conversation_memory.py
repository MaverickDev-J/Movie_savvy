# """Session‑only short‑term conversation memory."""
# from typing import List, Dict

# class ConversationMemory:
#     def __init__(self, max_turns: int = 10):
#         self.max_turns = max_turns
#         self.history: List[Dict[str, str]] = []  # [{"query": str, "response": str}, ...]

#     # ------------------------------------------------------------------
#     def add_turn(self, query: str, response: str) -> None:
#         """Append a new (query, response) pair and trim to the last *max_turns*."""
#         self.history.append({"query": query, "response": response})
#         if len(self.history) > self.max_turns:
#             self.history = self.history[-self.max_turns :]

#     def get_last_n(self, n: int) -> List[Dict[str, str]]:
#         """Return the last *n* turns (oldest ➜ newest)."""
#         return self.history[-n:]


# """Session-only short-term conversation memory with enhanced context management."""
# from typing import List, Dict, Optional
# import time

# class ConversationMemory:
#     def __init__(self, max_turns: int = 10, max_age_seconds: Optional[int] = None):
#         """
#         Initialize the conversation memory.
        
#         Args:
#             max_turns (int): Maximum number of turns to store (default: 10).
#             max_age_seconds (int, optional): Maximum age of a turn in seconds before it expires.
#         """
#         self.max_turns = max_turns
#         self.max_age_seconds = max_age_seconds
#         self.history: List[Dict[str, str]] = []  # [{"query": str, "response": str, "timestamp": float}, ...]

#     def add_turn(self, query: str, response: str) -> None:
#         """
#         Append a new (query, response) pair with a timestamp and manage history.
        
#         Args:
#             query (str): The user's query.
#             response (str): The system's response.
#         """
#         # Add new turn with current timestamp
#         turn = {"query": query, "response": response, "timestamp": time.time()}
#         self.history.append(turn)
        
#         # Trim history based on max_turns and max_age_seconds
#         self._trim_history()

#     def _trim_history(self) -> None:
#         """Remove old turns based on max_turns and max_age_seconds."""
#         # Remove expired turns if max_age_seconds is set
#         if self.max_age_seconds is not None:
#             current_time = time.time()
#             self.history = [
#                 turn for turn in self.history
#                 if (current_time - turn["timestamp"]) <= self.max_age_seconds
#             ]
        
#         # Trim to max_turns if exceeded
#         if len(self.history) > self.max_turns:
#             self.history = self.history[-self.max_turns:]

#     def get_last_n(self, n: int) -> List[Dict[str, str]]:
#         """
#         Return the last n turns (oldest to newest).
        
#         Args:
#             n (int): Number of turns to retrieve.
        
#         Returns:
#             List of turn dictionaries.
#         """
#         return self.history[-n:]

#     def get_context(self, max_chars: int = 2000) -> str:
#         """
#         Generate a formatted string of the conversation history for prompt inclusion.
        
#         Args:
#             max_chars (int): Maximum character length of the context string.
        
#         Returns:
#             str: Formatted conversation history.
#         """
#         context_lines = []
#         total_chars = 0
        
#         # Iterate from newest to oldest
#         for turn in reversed(self.history):
#             turn_text = f"User: {turn['query']}\nAI: {turn['response']}\n"
#             turn_chars = len(turn_text)
            
#             if total_chars + turn_chars > max_chars:
#                 break
                
#             context_lines.append(turn_text)
#             total_chars += turn_chars
        
#         # Reverse back to chronological order (oldest first)
#         return "".join(reversed(context_lines)).strip() or "(no prior conversation)"

#     def clear(self) -> None:
#         """Clear the conversation history."""
#         self.history = []












"""Session-only short-term conversation memory with enhanced context management."""
from typing import List, Dict, Optional
import time

class ConversationMemory:
    def __init__(self, max_turns: int = 10, max_age_seconds: Optional[int] = None):
        """
        Initialize the conversation memory.
        
        Args:
            max_turns (int): Maximum number of turns to store (default: 10).
            max_age_seconds (int, optional): Maximum age of a turn in seconds before it expires.
        """
        self.max_turns = max_turns
        self.max_age_seconds = max_age_seconds
        self.history: List[Dict[str, str]] = []  # [{"query": str, "response": str, "timestamp": float}, ...]

    def add_turn(self, query: str, response: str) -> None:
        """
        Append a new (query, response) pair with a timestamp and manage history.
        
        Args:
            query (str): The user's query.
            response (str): The system's response.
        """
        # Add new turn with current timestamp
        turn = {"query": query, "response": response, "timestamp": time.time()}
        self.history.append(turn)
        
        # Trim history based on max_turns and max_age_seconds
        self._trim_history()

    def _trim_history(self) -> None:
        """Remove old turns based on max_turns and max_age_seconds."""
        # Remove expired turns if max_age_seconds is set
        if self.max_age_seconds is not None:
            current_time = time.time()
            self.history = [
                turn for turn in self.history
                if (current_time - turn["timestamp"]) <= self.max_age_seconds
            ]
        
        # Trim to max_turns if exceeded
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_last_n(self, n: int) -> List[Dict[str, str]]:
        """
        Return the last n turns (oldest to newest).
        
        Args:
            n (int): Number of turns to retrieve.
        
        Returns:
            List of turn dictionaries.
        """
        return self.history[-n:]

    def get_context(self, max_chars: int = 2000) -> str:
        """
        Generate a formatted string of the conversation history for prompt inclusion, 
        aligned with Alpaca prompt style.
        
        Args:
            max_chars (int): Maximum character length of the context string.
        
        Returns:
            str: Formatted conversation history.
        """
        context_lines = []
        total_chars = 0
        
        # Iterate from newest to oldest
        for turn in reversed(self.history):
            turn_text = f"Instruction: {turn['query']}\nOutput: {turn['response']}\n"
            turn_chars = len(turn_text)
            
            if total_chars + turn_chars > max_chars:
                break
                
            context_lines.append(turn_text)
            total_chars += turn_chars
        
        # Reverse back to chronological order (oldest first)
        return "".join(reversed(context_lines)).strip() or "(no prior conversation)"
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.history = []