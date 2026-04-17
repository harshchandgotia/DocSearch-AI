import logging

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from nodes import format_conversation_context
from prompts.prompts import DIRECT_GENERATE_PROMPT

logger = logging.getLogger(__name__)


def direct_generate(state: SelfRAGState, llm, app_config: dict) -> dict:
    conversation_context = format_conversation_context(
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
        return {
            "answer": response.content,
            "context": "",
            "retrieval_used": False,
        }
    except Exception:
        logger.exception("Direct generation failed.")
        return {
            "answer": "I encountered an error while generating the answer.",
            "context": "",
            "retrieval_used": False,
        }

