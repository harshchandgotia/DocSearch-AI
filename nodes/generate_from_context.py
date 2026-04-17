import logging

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from nodes import format_conversation_context
from prompts.prompts import GENERATE_FROM_CONTEXT_PROMPT

logger = logging.getLogger(__name__)


def generate_from_context(state: SelfRAGState, llm, app_config: dict) -> dict:
    relevant_docs = state.get("relevant_docs", [])
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    conv = format_conversation_context(state.get("conversation_history", []))
    conversation_context = (
        conv + "\n\nNow answer the following based only on the provided documents:"
        if conv else ""
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
        logger.exception("Generation from context failed.")
        return {
            "answer": "I encountered an error while generating the answer.",
            "context": context,
        }

