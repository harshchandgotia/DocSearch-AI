from typing import TypedDict


class SelfRAGState(TypedDict):
    question: str
    original_question: str
    conversation_history: list[dict]
    pinned_pdf_ids: list[str]
    needs_retrieval: bool
    retrieval_query: str
    documents: list
    relevant_docs: list
    context: str
    answer: str
    is_supported: str
    is_useful: str
    revision_count: int
    rewrite_count: int
    sources: list[str]
    retrieval_used: bool
    no_answer: bool
