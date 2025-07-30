import pandas as pd
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

import config

#def get_retrievers(df):
#    embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
#    documents = get_documents(df)
#    # Initialize the embedding model (E5 large). `normalize_embeddings=True` to use cosine similarity.#
#
#    # Create a FAISS vector store from the documents
#    vector_store = FAISS.from_documents(documents, embedding)
#    #vector_store = None
#
#    # Create a BM25 retriever from the same documents
#    bm25_retriever = BM25Retriever.from_documents(documents)
#    #bm25_retriever = None
#    return (vector_store, bm25_retriever)


def get_retrievers(documents):
    embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    # Initialize the embedding model (E5 large). `normalize_embeddings=True` to use cosine similarity.

    # Create a FAISS vector store from the documents
    vector_store = FAISS.from_documents(documents, embedding)
    #vector_store = None

    # Create a BM25 retriever from the same documents
    bm25_retriever = BM25Retriever.from_documents(documents)
    #bm25_retriever = None
    return (vector_store, bm25_retriever)