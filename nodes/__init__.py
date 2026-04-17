def format_conversation_context(conversation_history: list[dict]) -> str:
    """Format the last 6 conversation turns into a readable string for prompts."""
    if not conversation_history:
        return ""
    recent = conversation_history[-6:]
    lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent]
    return "Prior conversation:\n" + "\n".join(lines)
