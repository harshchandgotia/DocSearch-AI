from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import GENERATE_FROM_CONTEXT_PROMPT


def _format_conversation_context(conversation_history: list[dict]) -> str:
    if not conversation_history:
        return ""
    recent = conversation_history[-6:]
    lines = []
    for msg in recent:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "Prior conversation:\n" + "\n".join(lines) + "\n\nNow answer the following based only on the provided documents:"


def generate_from_context(state: SelfRAGState, llm, app_config: dict) -> dict:
    relevant_docs = state.get("relevant_docs", [])
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    conversation_context = _format_conversation_context(
        state.get("conversation_history", [])
    )

    try:
        prompt = GENERATE_FROM_CONTEXT_PROMPT.format(
            conversation_context=conversation_context,
            context=context,
            question=state["question"],
        )
        response = llm.invoke([
            SystemMessage(content="You are a precise question-answering assistant."),
            HumanMessage(content=prompt),
        ])
        return {"answer": response.content, "context": context}
    except Exception:
        return {
            "answer": "I encountered an error while generating the answer.",
            "context": context,
        }
