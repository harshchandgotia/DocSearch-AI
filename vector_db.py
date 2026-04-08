# Importing dependencies
import os
from dotenv import load_dotenv
from chunker import text_splitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()


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


def process_questions(index_tuple, pdf_id, question):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'filter': {'pdf_id': pdf_id}})
    )
    return index_tuple, qa.invoke(question)["result"]


# Global initialization of models and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=128
)

index = create_pineconeInstance()
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)


def ingest_document(extracted_text: str, pdf_id: str):
    chunks = text_splitter.split_text(extracted_text)
    documents = [Document(page_content=text, metadata={"pdf_id": pdf_id}) for text in chunks]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)


def retrieval(questions):
    # questions is a list of { "pdf_id": str, "questions": [str] }
    # Flatten into (doc_index, q_index, pdf_id, question) tuples for concurrent execution
    tasks = []
    for doc_index, item in enumerate(questions):
        pdf_id = item["pdf_id"]
        for q_index, q_text in enumerate(item["questions"]):
            tasks.append((doc_index, q_index, pdf_id, q_text))

    raw_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_questions, (doc_index, q_index), pdf_id, q_text): (doc_index, q_index)
            for doc_index, q_index, pdf_id, q_text in tasks
        }
        for future in as_completed(futures):
            (doc_index, q_index), answer = future.result()
            raw_results.append((doc_index, q_index, answer))

    # Sort by doc_index then q_index to restore original order
    raw_results.sort(key=lambda x: (x[0], x[1]))

    # Reconstruct grouped response mirroring input structure
    response = []
    for doc_index, item in enumerate(questions):
        answers_for_doc = [
            answer
            for d_idx, q_idx, answer in raw_results
            if d_idx == doc_index
        ]
        response.append({
            "pdf_id": item["pdf_id"],
            "answers": answers_for_doc
        })

    return response


def delete_document(pdf_id: str):
    index.delete(filter={"pdf_id": pdf_id})
