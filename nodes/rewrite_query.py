import logging

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import REWRITE_QUERY_PROMPT

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    retrieval_query: str


def rewrite_query(state: SelfRAGState, llm, app_config: dict) -> dict:
    try:
        prompt = REWRITE_QUERY_PROMPT.format(
            question=state["question"],
        )
        structured_llm = llm.with_structured_output(RewriteResult)
        result = structured_llm.invoke([
            SystemMessage(content="You are a query optimization system."),
            HumanMessage(content=prompt),
        ])
        return {
            "retrieval_query": result.retrieval_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }
    except Exception:
        logger.exception("Query rewrite failed; keeping original query.")
        # Fall back to the original question so the next retrieve isn't a no-op
        return {
            "retrieval_query": state.get("retrieval_query") or state["question"],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

