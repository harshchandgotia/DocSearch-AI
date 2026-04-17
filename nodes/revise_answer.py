import logging

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import REVISE_ANSWER_PROMPT

logger = logging.getLogger(__name__)


def revise_answer(state: SelfRAGState, llm, app_config: dict) -> dict:
    try:
        prompt = REVISE_ANSWER_PROMPT.format(
            context=state.get("context", ""),
            answer=state.get("answer", ""),
        )
        response = llm.invoke([
            SystemMessage(content="You are an answer revision system."),
            HumanMessage(content=prompt),
        ])
        return {
            "answer": response.content,
            "revision_count": state.get("revision_count", 0) + 1,
        }
    except Exception:
        logger.exception("Answer revision failed; keeping original answer.")
        return {
            "answer": state.get("answer", ""),
            "revision_count": state.get("revision_count", 0) + 1,
        }

