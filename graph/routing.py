from config import load_config
from graph.state import SelfRAGState

_config = load_config()
_max_revision_retries = _config["self_rag"]["max_revision_retries"]
_max_rewrite_retries = _config["self_rag"]["max_rewrite_retries"]


def route_after_retrieval_decision(state: SelfRAGState) -> str:
    if state.get("needs_retrieval"):
        return "retrieve"
    return "direct_generate"


def route_after_grading(state: SelfRAGState) -> str:
    if len(state.get("relevant_docs", [])) > 0:
        return "generate_from_context"
    return "no_docs_found"


def route_after_hallucination_check(state: SelfRAGState) -> str:
    if state.get("is_supported") == "fully_supported":
        return "check_usefulness"
    if state.get("revision_count", 0) < _max_revision_retries:
        return "revise_answer"
    return "no_docs_found"


def route_after_usefulness_check(state: SelfRAGState) -> str:
    if state.get("is_useful") == "useful":
        return "END"
    if state.get("rewrite_count", 0) < _max_rewrite_retries:
        return "rewrite_query"
    return "END"


def no_docs_found(state: SelfRAGState) -> dict:
    return {
        "answer": "I couldn't find a reliable answer in the provided documents.",
        "no_answer": True,
    }
