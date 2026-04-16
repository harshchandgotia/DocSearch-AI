from pydantic import BaseModel
from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import REWRITE_QUERY_PROMPT


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
        return {"rewrite_count": state.get("rewrite_count", 0) + 1}
