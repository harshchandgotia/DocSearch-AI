from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import DIRECT_GENERATE_PROMPT


def _format_conversation_context(conversation_history: list[dict]) -> str:
    if not conversation_history:
        return ""
    recent = conversation_history[-6:]
    lines = []
    for msg in recent:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "Prior conversation:\n" + "\n".join(lines)


def direct_generate(state: SelfRAGState, llm, app_config: dict) -> dict:
    conversation_context = _format_conversation_context(
        state.get("conversation_history", [])
    )

    try:
        prompt = DIRECT_GENERATE_PROMPT.format(
            conversation_context=conversation_context,
            question=state["question"],
        )
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt),
        ])
        return {"answer": response.content}
    except Exception:
        return {"answer": "I encountered an error while generating the answer."}
