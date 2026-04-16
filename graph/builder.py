import functools

from langgraph.graph import StateGraph, START, END

from graph.state import SelfRAGState
from graph.routing import (
    route_after_retrieval_decision,
    route_after_grading,
    route_after_hallucination_check,
    route_after_usefulness_check,
    no_docs_found,
)
from nodes.decide_retrieval import decide_retrieval
from nodes.retrieve import retrieve
from nodes.grade_documents import grade_documents
from nodes.generate_from_context import generate_from_context
from nodes.direct_generate import direct_generate
from nodes.check_hallucination import check_hallucination
from nodes.revise_answer import revise_answer
from nodes.check_usefulness import check_usefulness
from nodes.rewrite_query import rewrite_query


def build_graph(config: dict, llm, vector_store):
    graph = StateGraph(SelfRAGState)

    # Bind llm and config to all nodes via functools.partial
    graph.add_node("decide_retrieval", functools.partial(decide_retrieval, llm=llm, app_config=config))
    graph.add_node("retrieve", functools.partial(retrieve, llm=llm, app_config=config, vector_store=vector_store))
    graph.add_node("grade_documents", functools.partial(grade_documents, llm=llm, app_config=config))
    graph.add_node("generate_from_context", functools.partial(generate_from_context, llm=llm, app_config=config))
    graph.add_node("direct_generate", functools.partial(direct_generate, llm=llm, app_config=config))
    graph.add_node("check_hallucination", functools.partial(check_hallucination, llm=llm, app_config=config))
    graph.add_node("revise_answer", functools.partial(revise_answer, llm=llm, app_config=config))
    graph.add_node("check_usefulness", functools.partial(check_usefulness, llm=llm, app_config=config))
    graph.add_node("rewrite_query", functools.partial(rewrite_query, llm=llm, app_config=config))
    graph.add_node("no_docs_found", no_docs_found)

    # Wire edges
    graph.add_edge(START, "decide_retrieval")

    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_retrieval_decision,
        {"retrieve": "retrieve", "direct_generate": "direct_generate"},
    )

    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate_from_context": "generate_from_context", "no_docs_found": "no_docs_found"},
    )

    graph.add_edge("generate_from_context", "check_hallucination")

    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {"check_usefulness": "check_usefulness", "revise_answer": "revise_answer", "no_docs_found": "no_docs_found"},
    )

    graph.add_edge("revise_answer", "check_hallucination")

    graph.add_conditional_edges(
        "check_usefulness",
        route_after_usefulness_check,
        {"END": END, "rewrite_query": "rewrite_query"},
    )

    graph.add_edge("rewrite_query", "retrieve")

    graph.add_edge("direct_generate", END)
    graph.add_edge("no_docs_found", END)

    return graph.compile()
