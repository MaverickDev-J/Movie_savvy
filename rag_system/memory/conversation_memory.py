"""Session‑only short‑term conversation memory."""
from typing import List, Dict

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []  # [{"query": str, "response": str}, ...]

    # ------------------------------------------------------------------
    def add_turn(self, query: str, response: str) -> None:
        """Append a new (query, response) pair and trim to the last *max_turns*."""
        self.history.append({"query": query, "response": response})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def get_last_n(self, n: int) -> List[Dict[str, str]]:
        """Return the last *n* turns (oldest ➜ newest)."""
        return self.history[-n:]