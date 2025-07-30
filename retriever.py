from typing import List, Any, Optional, Dict, Tuple ,TypedDict, Annotated
import os
import pickle
import torch

from langchain_community.document_loaders import NotionDBLoader
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.docstore.document import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.storage import InMemoryByteStore


from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition

from langchain_community.vectorstores import FAISS

from palimpsest import Palimpsest
from teamly_retriever import (
    TeamlyRetriever, 
    TeamlyContextualCompressionRetriever
)

import config

def load_vectorstore(file_path: str, embedding_model_name: str) -> FAISS:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No vectorstore found at {file_path}")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    return FAISS.load_local(
        file_path, embeddings, allow_dangerous_deserialization=True
    )


def get_retriever_multi():
    notion_vs = load_vectorstore(config.NOTION_INDEX_FOLDER, config.EMBEDDING_MODEL)
    chats_vs = load_vectorstore(config.CHATS_INDEX_FOLDER, config.EMBEDDING_MODEL)
    k = 5
    ensemble = EnsembleRetriever(
        retrievers=[notion_vs.as_retriever(search_kwargs={"k": k}), chats_vs.as_retriever(search_kwargs={"k": k})],
        weights=[0.5, 0.5]                  # adjust to favor text vs. images
    )
    reranker_model = HuggingFaceCrossEncoder(model_name=config.RERANKING_MODEL)
    reranker = CrossEncoderReranker(model=reranker_model, top_n=3)
    retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble
            )

    def search(query: str) -> List[Document]:
        result = retriever.invoke(query, search_kwargs={"k": k})
        #docs = retriever.similarity_search_with_score(query, k=5)
        #result = [doc for doc, score in docs if score >= 0.20]
        return result
    return search


def get_retriever_plain():
    #Load document store from persisted storage
    #loading list of problem numbers as ids
    
    vector_store_path = config.ASSISTANT_INDEX_FOLDER
    vectorstore = load_vectorstore(vector_store_path, config.EMBEDDING_MODEL)
    reranker_model = HuggingFaceCrossEncoder(model_name=config.RERANKING_MODEL)
    reranker = CrossEncoderReranker(model=reranker_model, top_n=3)
    with open(f'{vector_store_path}/docstore.pkl', 'rb') as file:
        documents = pickle.load(file)

    doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]
    store = InMemoryByteStore()
    id_key = "problem_number"
    MAX_RETRIEVALS = 5
    multi_retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": MAX_RETRIEVALS},
        )
    multi_retriever.docstore.mset(list(zip(doc_ids, documents)))
    retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=multi_retriever
            )

    def search(query: str) -> List[Document]:
        result = retriever.invoke(query, search_kwargs={"k": MAX_RETRIEVALS})
        return result
    return search

def get_retriever_teamly():
    MAX_RETRIEVALS = 3

    teamly_retriever = TeamlyRetriever("./auth.json", k=30)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    reranker_model = HuggingFaceCrossEncoder(model_name=config.RERANKING_MODEL, model_kwargs = {'trust_remote_code': True, "device": device})
    reranker = CrossEncoderReranker(model=reranker_model, top_n=MAX_RETRIEVALS)
    retriever = TeamlyContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=teamly_retriever
            )

    def search(query: str) -> List[Document]:
        result = retriever.invoke(query, search_kwargs={"k": MAX_RETRIEVALS})
        #torch.cuda.empty_cache()
        return result

    return search


def get_retriever_faiss():
    MAX_RETRIEVALS = 3

    vector_store_path = config.ASSISTANT_INDEX_FOLDER
    vectorstore = load_vectorstore(vector_store_path, config.EMBEDDING_MODEL)
    with open(f'{vector_store_path}/docstore.pkl', 'rb') as file:
        documents = pickle.load(file)

    doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]
    store = InMemoryByteStore()
    id_key = "problem_number"
    multi_retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": MAX_RETRIEVALS},
        )
    multi_retriever.docstore.mset(list(zip(doc_ids, documents)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_model = HuggingFaceCrossEncoder(model_name=config.RERANKING_MODEL, model_kwargs = {'trust_remote_code': True, "device": device})
    reranker = CrossEncoderReranker(model=reranker_model, top_n=MAX_RETRIEVALS)
    retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=multi_retriever
            )

    def search(query: str) -> List[Document]:
        result = retriever.invoke(query, search_kwargs={"k": MAX_RETRIEVALS})
        return result

    return search


def get_retriever():
    retriever_type = config.RETRIEVER_TYPE
    if retriever_type == "teamly":
        return get_retriever_teamly()
    return get_retriever_faiss()

search = get_retriever()

def get_search_tool(anonymizer: Palimpsest = None):
    
    @tool
    def search_kb(query: str) -> str:
        """Retrieves from knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question
        Returns:
            Context from knowledgebase suitable for the query.
        """
        
        found_docs = search(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return search_kb

if __name__ == '__main__':
    search_kb = get_search_tool()
    #answer = search_kb("Кто такие кей юзеры?")
    #print(answer)

    answer = search_kb("Что делать, если не пришёл ЗОР?")
    print(answer)

    answer = search_kb("Что делать, если не пришёл ЗОР?")
    print(answer)

