"""Utility to build the final prompt that includes memory + docs + question."""

def build_prompt(memory_history: str, retrieved_docs: str, current_query: str) -> str:
    template = (
        "You are an AI assistant. Use the conversation history and the supporting documents below to answer the question.\n\n"
        "Conversation history:\n{memory}\n\n"
        "Relevant documents:\n{docs}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
    return template.format(
        memory=memory_history.strip() or "(no prior conversation)",
        docs=retrieved_docs.strip() or "(no supporting documents)",
        query=current_query.strip(),
    )