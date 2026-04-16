from pydantic import BaseModel
from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import RETRIEVAL_DECISION_PROMPT


class RetrievalDecision(BaseModel):
    retrieve: bool
    reason: str


def _format_history(conversation_history: list[dict]) -> str:
    if not conversation_history:
        return "No prior conversation."
    lines = []
    for msg in conversation_history[-6:]:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def decide_retrieval(state: SelfRAGState, llm, app_config: dict) -> dict:
    try:
        history_str = _format_history(state.get("conversation_history", []))
        prompt = RETRIEVAL_DECISION_PROMPT.format(
            conversation_history=history_str,
            question=state["question"],
        )
        structured_llm = llm.with_structured_output(RetrievalDecision)
        result = structured_llm.invoke([
            SystemMessage(content="You are a retrieval decision system."),
            HumanMessage(content=prompt),
        ])
        return {
            "needs_retrieval": result.retrieve,
            "retrieval_used": result.retrieve,
        }
    except Exception:
        return {"needs_retrieval": True, "retrieval_used": True}
