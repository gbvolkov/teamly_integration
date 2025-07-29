import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from build_documents import get_documents
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", encode_kwargs={"normalize_embeddings": True})


def get_retrievers(df):
    documents = get_documents(df)
    # Initialize the embedding model (E5 large). `normalize_embeddings=True` to use cosine similarity.

    # Create a FAISS vector store from the documents
    vector_store = FAISS.from_documents(documents, embedding)

    # Create a BM25 retriever from the same documents
    bm25_retriever = BM25Retriever.from_documents(documents)
    return (vector_store, bm25_retriever)