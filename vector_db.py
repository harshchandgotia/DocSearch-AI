import os
from dotenv import load_dotenv
from chunker import text_splitter
from langchain_classic.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from config import load_config
from graph.builder import build_graph

load_dotenv()

config = load_config()


def create_pineconeInstance():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = os.getenv("PINECONE_INDEX_NAME", "docsearch-ai")

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 output dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(index_name)
    return index


# Global initialization of models and vector store
embedding_model = HuggingFaceEmbeddings(model_name=config["embeddings"]["model"])

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=config["llm"]["groq_model"],
    temperature=config["llm"]["temperature"],
    max_tokens=config["llm"]["max_tokens"],
)

index = create_pineconeInstance()
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

# Build Self-RAG graph once at module load
self_rag_graph = build_graph(config, llm, vector_store)


def ingest_document(extracted_text: str, pdf_id: str):
    chunks = text_splitter.split_text(extracted_text)
    documents = [Document(page_content=text, metadata={"pdf_id": pdf_id}) for text in chunks]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)


def run_query(question: str, pinned_pdf_ids: list[str], conversation_history: list[dict]) -> dict:
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

    result = self_rag_graph.invoke(initial_state)

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
    index.delete(filter={"pdf_id": pdf_id})
