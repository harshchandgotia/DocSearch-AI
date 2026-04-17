import logging
from typing import Literal

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import RETRIEVAL_DECISION_PROMPT

logger = logging.getLogger(__name__)


class RetrievalDecision(BaseModel):
    retrieve: Literal["yes", "no"]
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
        needs_retrieval = result.retrieve == "yes"
        return {
            "needs_retrieval": needs_retrieval,
            "retrieval_used": needs_retrieval,
        }
    except Exception:
        logger.exception("Retrieval decision failed; defaulting to retrieval.")
        return {"needs_retrieval": True, "retrieval_used": True}

