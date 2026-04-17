import os
from dotenv import load_dotenv
from chunker import text_splitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from langchain_groq import ChatGroq
from config import load_config
from graph.builder import build_graph

load_dotenv()

config = load_config()

# Lazily initialized singletons
_embedding_model = None
_llm = None
_index = None
_vector_store = None
_self_rag_graph = None


def _create_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "docsearch-ai")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 output dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def _init():
    """Initialize all singletons on first use."""
    global _embedding_model, _llm, _index, _vector_store, _self_rag_graph
    if _self_rag_graph is not None:
        return
    _embedding_model = HuggingFaceEmbeddings(model_name=config["embeddings"]["model"])
    _llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=config["llm"]["groq_model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )
    _index = _create_pinecone_index()
    _vector_store = PineconeVectorStore(index=_index, embedding=_embedding_model)
    _self_rag_graph = build_graph(config, _llm, _vector_store)


def ingest_document(extracted_text: str, pdf_id: str):
    _init()
    chunks = text_splitter.split_text(extracted_text)
    documents = [Document(page_content=text, metadata={"pdf_id": pdf_id}) for text in chunks]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    _vector_store.add_documents(documents=documents, ids=uuids)


def run_query(question: str, pinned_pdf_ids: list[str], conversation_history: list[dict]) -> dict:
    _init()
    initial_state = {
        "question": question,
        "original_question": question,
        "conversation_history": conversation_history,
        "pinned_pdf_ids": pinned_pdf_ids,
        "needs_retrieval": False,
        "retrieval_query": question,
        "documents": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "is_supported": "",
        "is_useful": "",
        "revision_count": 0,
        "rewrite_count": 0,
        "sources": [],
        "retrieval_used": False,
        "no_answer": False,
    }
    result = _self_rag_graph.invoke(initial_state)
    return {
        "answer": result.get("answer", ""),
        "is_supported": result.get("is_supported", ""),
        "is_useful": result.get("is_useful", ""),
        "revision_count": result.get("revision_count", 0),
        "rewrite_count": result.get("rewrite_count", 0),
        "sources": result.get("sources", []),
        "retrieval_used": result.get("retrieval_used", False),
        "no_answer": result.get("no_answer", False),
    }


def delete_document(pdf_id: str):
    _init()
    _index.delete(filter={"pdf_id": pdf_id})
