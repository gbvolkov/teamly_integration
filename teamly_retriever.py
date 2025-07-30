from __future__ import annotations


from typing import List, Optional, Dict


from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain.schema import Document
from langchain.schema.runnable import RunnableConfig
from langchain.schema.retriever import BaseRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from build_index import get_retrievers
from teamly_api_wrapper import TeamlyAPIWrapper

import config

# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class TeamlyRetriever(BaseRetriever, TeamlyAPIWrapper):
    """
    LangChain-compatible wrapper around Teamly semantic search.

    Required auth/connection data are read from *auth_data_store* – the same
    JSON structure you used before:

    ```json
    {
      "base_url":      "...",
      "client_id":     "...",
      "client_secret": "...",
      "auth_code":     "...",   # either refresh-token or one-shot code
      "redirect_uri":  "..."
    }
    ```
    """
    idx_vectors: FAISS = None
    idx_bm25: BM25Retriever = None
    
    # pydantic-style model config so arbitrary attrs are allowed
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    # ---------------------------------------------------------------------
    # Construction / auth helpers
    # ---------------------------------------------------------------------

    def __init__(self, auth_data_store: str, *, k: int = 10) -> None:
        """
        Parameters
        ----------
        auth_data_store:
            Path to the JSON file that stores credentials and the last
            (refresh) token. The file will be updated in-place whenever a new
            token is issued.
        k:
            How many documents to return for every query.
        """
        BaseRetriever.__init__(self)
        TeamlyAPIWrapper.__init__(self, auth_data_store = auth_data_store)
        self.k = k

        #shall be last call in chain, since it uses initialize teamly search params
        self.load_sd_articles_index()

    def load_sd_articles_index(self):
        (self.idx_vectors, self.idx_bm25) = get_retrievers(self.sd_documents)            

    def _get_documents_from_teamly_search(self, query: str) -> List[Document]:
        """Synchronous retrieval used by most LC components."""
        raw_hits = self._semantic_search(query)[: self.k]
        
        # Group hits by space_id and article_id
        grouped_hits = {}
        for hit in raw_hits:
            key = (hit["space_id"], hit["article_id"])
            if key not in grouped_hits:
                grouped_hits[key] = []
            grouped_hits[key].append(hit)
        
        # Merge hits with same space_id and article_id
        documents = []
        for hits in grouped_hits.values():
            if len(hits) == 1:
                documents.append(self._to_document(hits[0]))
            else:
                # Merge multiple hits into one document
                hits = sorted(hits, key=lambda x: x["offset"])

                merged_text = "\n\n".join(hit["text"] for hit in hits)
                max_score = max(hit["score"] for hit in hits)
                total_token_length = sum(hit["chunk_token_length"] for hit in hits)
                total_length = sum(hit["length"] for hit in hits)
                offsets = [hit["offset"] for hit in hits]
                
                # Use the first hit as base for metadata
                base_hit = hits[0]
                document = Document(
                    page_content=f"{merged_text}\n\nСсылка на статью:https://kb.ileasing.ru/space/{base_hit['space_id']}/article/{base_hit['article_id']}",
                    metadata={
                        "docid": f"{base_hit['space_id']}_{base_hit['article_id']}_merged",
                        "space_id": base_hit["space_id"],
                        "article_id": base_hit["article_id"],
                        "article_title": base_hit["article_title"],
                        "score": max_score,
                        "chunk_token_length": total_token_length,
                        "offset": offsets,
                        "length": total_length,
                        "merged": True,  # Flag indicating this is a merged document
                        "original_hits_count": len(hits),  # How many hits were merged
                    },
                )
                documents.append(document)
        return documents

    def _get_documents_from_sd_tables(self, query: str) -> List[Document]:
        documents = []
        if self.idx_vectors:
            v_res = self.idx_vectors.similarity_search(query, k=5)
            documents.extend(v_res)
        if self.idx_bm25:
            bm25_res = self.idx_bm25.invoke(query)[:5]
            documents.extend(bm25_res)
        return documents
        
    # --------------------------------- public LangChain hook -------------
    def _get_relevant_documents(        # type: ignore[override]
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,  # for LC internal plumbing
    ) -> List[Document]:
        documents = self._get_documents_from_teamly_search(query = query)
        sd_documents = self._get_documents_from_sd_tables(query = query)
        #documents.extend(sd_documents)
        return documents
    
    async def _aget_relevant_documents(  # type: ignore[override]
        self,
        query: str,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """Naïve asyncio wrapper – uses thread pool because requests is sync."""
        from asyncio import get_running_loop
        return await get_running_loop().run_in_executor(
            None, lambda: self.get_relevant_documents(query=query)
        )


class TeamlyContextualCompressionRetriever(ContextualCompressionRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        documents = super()._get_relevant_documents(query, run_manager=run_manager, **kwargs)
        if isinstance(self.base_retriever, TeamlyRetriever):
            for doc in documents:
                article_id = doc.metadata.get("article_id")
                space_id = doc.metadata.get("space_id", "")
                if article_id:
                    doc.page_content = f"{self.base_retriever.get_article(article_id)}\n\nСсылка на статью: {self.base_retriever.base_url}/space/{space_id}/article/{article_id}"
        return documents

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
    from typing import Any
    from pprint import pprint

    class KBDocumentPromptTemplate(StringPromptTemplate):
        max_length : int = 0
        def __init__(self, max_length: int, **kwargs: Any):
            super().__init__(**kwargs)
            self.max_length = max_length

        def format(self, **kwargs: Any) -> str:
            page_content = kwargs.pop("page_content")
            #problem_number = kwargs.pop("problem_number")
            #chunk_size = kwargs.pop("actual_chunk_size")
            #here additional data could be retrieved based on problem_number
            result = page_content
            if self.max_length > 0:
                result = result[:self.max_length]
            return result

        @property
        def _prompt_type(self) -> str:
            return "kb_document"
        
    retriever = TeamlyRetriever(auth_data_store="./auth.json", k=5)

    llm = ChatOpenAI(model="gpt-4.1-mini")
    with open("./prompt.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_txt),
                ("human", "User request: \n{input}\n\nContext: \n{context}"),
            ]
        )

    my_prompt = KBDocumentPromptTemplate(0, input_variables=["page_content", "article_id", "article_title"])

    docs_chain = create_stuff_documents_chain(llm, system_prompt, document_prompt=my_prompt, document_separator='\n#EOD\n\n')
    rag_chain = create_retrieval_chain(retriever, docs_chain)

    result = rag_chain.invoke({"input": "Кто такие key users?"})
    pprint(result)