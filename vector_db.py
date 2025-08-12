# Importing dependencies
import os
from dotenv import load_dotenv
from chunker import text_splitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# loading environment variable
load_dotenv()
# Initializing pinecone instance
def create_pineconeInstance():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "hackrx6"

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # For all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    # Connect to index
    index = pc.Index(index_name)
    return index

def process_questions(question: str, qa):
    index, q = question
    return index, qa.invoke(q)["result"]
# global initialization of the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Creating the vector store and llm inferencing
def generate_response(extracted_text: str, questions: list[str]):
    chunks = text_splitter.split_text(extracted_text)
    documents = [Document(page_content=text) for text in chunks]

    index = create_pineconeInstance()
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    # Running inference
    llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=128
    )
    # model = ChatHuggingFace(llm=llm)
    qa = RetrievalQA.from_chain_type(  
        llm=llm,  
        chain_type="stuff",  
        retriever=vector_store.as_retriever()  
    )
    response = []
    # for question in questions:
    #     sample = qa.invoke(question)
    #     print(sample)
    #     response.append(qa.invoke(question)["result"])
    # return response
    question_tuple = list(enumerate(questions))
    # [[0, q1], [1, q2]]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_questions, question, qa) for question in question_tuple]
        for future in as_completed(futures):
            response.append((future.result()))
    response.sort(key = lambda x: x[0])
    final_response = [answer[1] for answer in response]
    return final_response

