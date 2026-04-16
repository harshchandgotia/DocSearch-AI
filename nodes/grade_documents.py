from pydantic import BaseModel
from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import DOCUMENT_GRADING_PROMPT


class RelevanceGrade(BaseModel):
    is_relevant: bool


def grade_documents(state: SelfRAGState, llm, app_config: dict) -> dict:
    documents = state.get("documents", [])
    relevant_docs = []
    sources = []

    structured_llm = llm.with_structured_output(RelevanceGrade)

    for doc in documents:
        try:
            prompt = DOCUMENT_GRADING_PROMPT.format(
                question=state["question"],
                document_content=doc.page_content,
            )
            result = structured_llm.invoke([
                SystemMessage(content="You are a relevance grading system."),
                HumanMessage(content=prompt),
            ])
            if result.is_relevant:
                relevant_docs.append(doc)
                pdf_id = doc.metadata.get("pdf_id", "unknown")
                if pdf_id not in sources:
                    sources.append(pdf_id)
        except Exception:
            relevant_docs.append(doc)
            pdf_id = doc.metadata.get("pdf_id", "unknown")
            if pdf_id not in sources:
                sources.append(pdf_id)

    return {"relevant_docs": relevant_docs, "sources": sources}
