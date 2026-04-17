import logging

from graph.state import SelfRAGState

logger = logging.getLogger(__name__)


def retrieve(state: SelfRAGState, llm, app_config: dict, vector_store) -> dict:
    k = app_config["self_rag"]["retrieval_k"]
    pinned_pdf_ids = state.get("pinned_pdf_ids", [])
    query = state.get("retrieval_query", state["question"])

    if not pinned_pdf_ids:
        logger.warning("No pinned_pdf_ids provided; retrieval returning empty results.")
        return {"documents": [], "retrieval_used": True}

    if len(pinned_pdf_ids) == 1:
        search_filter = {"pdf_id": {"$eq": pinned_pdf_ids[0]}}
    else:
        search_filter = {"pdf_id": {"$in": pinned_pdf_ids}}

    documents = vector_store.similarity_search(
        query, k=k, filter=search_filter
    )

    logger.info("Retrieved %d documents for query: %.80s", len(documents), query)
    return {"documents": documents, "retrieval_used": True}

