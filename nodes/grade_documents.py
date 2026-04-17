import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import DOCUMENT_GRADING_PROMPT

logger = logging.getLogger(__name__)


class RelevanceGrade(BaseModel):
    is_relevant: Literal["yes", "no"]


def _grade_single_doc(structured_llm, question: str, doc, index: int):
    """Grade one document; returns (index, is_relevant, pdf_id)."""
    try:
        prompt = DOCUMENT_GRADING_PROMPT.format(
            question=question,
            document_content=doc.page_content,
        )
        result = structured_llm.invoke([
            SystemMessage(content="You are a relevance grading system."),
            HumanMessage(content=prompt),
        ])
        is_relevant_bool = result.is_relevant == "yes"
        return index, is_relevant_bool, doc.metadata.get("pdf_id", "unknown")
    except Exception:
        logger.exception("Error grading document at index %d", index)
        # On failure, include the document rather than silently drop it
        return index, True, doc.metadata.get("pdf_id", "unknown")


def grade_documents(state: SelfRAGState, llm, app_config: dict) -> dict:
    documents = state.get("documents", [])
    if not documents:
        return {"relevant_docs": [], "sources": []}

    # Use retrieval_query for grading so rewritten queries get matched properly
    grading_question = state.get("retrieval_query") or state["question"]
    structured_llm = llm.with_structured_output(RelevanceGrade)

    # Collect results indexed by position to preserve document order
    results = [None] * len(documents)

    with ThreadPoolExecutor(max_workers=min(len(documents), 8)) as executor:
        futures = {
            executor.submit(_grade_single_doc, structured_llm, grading_question, doc, i): i
            for i, doc in enumerate(documents)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    relevant_docs = []
    sources = []
    for idx, is_relevant, pdf_id in results:
        if is_relevant:
            relevant_docs.append(documents[idx])
            if pdf_id not in sources:
                sources.append(pdf_id)

    return {"relevant_docs": relevant_docs, "sources": sources}

