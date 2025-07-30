from __future__ import annotations

import functools
import json
import requests
from typing import List, Optional, Dict

import pandas as pd

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

from load_table import get_data_from_json
from build_index import get_retrievers

import config

# ---------------------------------------------------------------------------
# Helper decorator – unchanged except for typing tweaks
# ---------------------------------------------------------------------------

def _authorization_wrapper(func):
    """Refreshes / exchanges the token if an API call returns 401."""
    @functools.wraps(func)
    def wrapper(self: "TeamlyRetriever", *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except requests.HTTPError as http_err:
            if http_err.response.status_code != 401:
                raise
            # token may have expired → refresh & retry once
            self.auth_code = self._get_token()
            return func(self, *args, **kwargs)
    return wrapper

def _get_article_text(base_url: str, article_info: Dict) -> str:
    """
    Traverse a ProseMirror-like editor JSON (embedded as a string inside the
    editorContentObject) and build a single text string where:

    • every node whose type is "text" contributes its text value  
    • every node (or mark) whose type is "url" *or* "link" contributes the URL  
    • every node whose type is "paragraph" contributes a newline after its children  

    The traversal is depth-first and keeps the document’s original order.
    """
    # 1. Parse the JSON that lives in editorContentObject["content"]
    raw_doc = article_info["editorContentObject"]["content"]
    doc = json.loads(raw_doc)

    pieces: List[str] = [article_info["title"], "\n"]

    def walk(nodes: Dict | list) -> None:
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            ntype = node.get("type")

            # ---  text nodes ----------------------------------------------------
            if ntype == "text":
                pieces.append(node.get("text", ""))

                # links may be attached as marks on a text node  -----------------
                for mark in node.get("marks", []):
                    if mark.get("type") in {"url", "link"}:
                        url_obj = mark.get("attrs", {}).get("link") or mark.get("attrs", {})
                        url_placement = url_obj.get("type", "external")
                        url = url_obj.get("url")
                        if url and isinstance(url, str):
                            #if url_placement == "internal":
                            #    url = f" {self.base_url}{url}"
                            if not url.startswith("https:") and not url.startswith("mailto:"):
                                url = base_url + url
                            pieces.append(f" {url}")
                    elif mark.get("type") in {"media"}:
                        url = mark.get("attrs", {}).get("src")
                        if url and isinstance(url, str) and not url.startswith("data:"):
                            if not url.startswith("https:"):
                                url = base_url + url
                            pieces.append(f" {url}")

            # ---  dedicated url / link nodes ------------------------------------
            elif ntype in {"url", "link"}:
                url_obj = node.get("attrs", {}).get("link") or node.get("attrs", {})
                url_placement = url_obj.get("type", "external")
                url = url_obj.get("url")
                if url and isinstance(url, str):
                    #if url_placement == "internal":
                    #    url = f" {self.base_url}{url}"
                    if not url.startswith("https:") and not url.startswith("mailto:"):
                        url = base_url + url
                    pieces.append(f" {url}")

            # ---  dedicated url / link nodes ------------------------------------
            elif ntype in {"media"}:
                url = node.get("attrs", {}).get("src") or node.get("attrs", {})
                if url and isinstance(url, str) and not url.startswith("data:"):
                    if not url.startswith("https:") and not url.startswith("mailto:"):
                        url = base_url + url
                    pieces.append(f" {url}")

            # ---  newline after a paragraph -------------------------------------
            elif ntype == "paragraph":
                pieces.append("\n")

            # ---  recurse into children -----------------------------------------
            for child in node.get("content", []):
                walk(child)


    # The root is usually a {"type": "doc", ...}
    walk(doc)
    return "".join(pieces)[:128000]


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class TeamlyRetriever(BaseRetriever):
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
    k: int = 10
    base_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    auth_code: str = ""
    redirect_uri: str = ""
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
        super().__init__()
        self.k = k

        self._auth_path = auth_data_store
        with open(auth_data_store, "r", encoding="utf-8") as f:
            self._auth_data = json.load(f)

        self.base_url: str = self._auth_data["base_url"].rstrip("/")
        self.client_id: str = self._auth_data["client_id"]
        self.client_secret: str = self._auth_data["client_secret"]
        self.auth_code: str = self._auth_data["auth_code"]  # may be refresh-token
        self.redirect_uri: str = self._auth_data["redirect_uri"]

        # dynamic header (gets rebuilt after refresh)
        self._headers = lambda: {
            "X-Account-Slug": "default",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_code}",
        }
        #shall be last call in chain, since it uses initialize teamly search params
        self.load_sd_articles_index()

    def load_sd_articles_index(self):
        
        with open ("./data/sd_articles.json", "r") as f:
            articles = json.load(f)
        df = pd.DataFrame()
        for article_id in articles["articles"]:
            article_info = self.get_article_info(article_id)
            space_id = article_info["space_id"]
            article_title = article_info["title"]
            raw_doc = article_info["editorContentObject"]["content"]
            doc = json.loads(raw_doc)
            arcticle_df = get_data_from_json(doc, space_id, article_id, article_title)
            df = pd.concat([df, arcticle_df], ignore_index=True)
        (self.idx_vectors, self.idx_bm25) = get_retrievers(df)            

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
        documents.extend(sd_documents)
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

    # ---------------------------------------------------------------------
    # Internal request helpers
    # ---------------------------------------------------------------------

    @_authorization_wrapper
    def _post(self, url: str, payload: dict) -> dict:
        resp = requests.post(f"{self.base_url}{url}", json=payload, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def _auth_post(self, url: str, payload: dict) -> str:
        # No bearer header during auth flow
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        refresh_token = data.get("refresh_token")
        access_token  = data.get("access_token")

        # Persist fresh refresh-token
        self.auth_code = refresh_token or access_token
        self._auth_data["auth_code"] = self.auth_code
        with open(self._auth_path, "w", encoding="utf-8") as fp:
            json.dump(self._auth_data, fp, ensure_ascii=False, indent=4)

        return access_token

    # token endpoints -----------------------------------------------------

    def _refresh_token(self) -> str:
        return self._auth_post(
            f"{self.base_url}/api/v1/auth/integration/refresh",
            {
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.auth_code,
            },
        )

    def _authorise(self) -> str:
        return self._auth_post(
            f"{self.base_url}/api/v1/auth/integration/authorize",
            {
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri":  self.redirect_uri,
                "code":          self.auth_code,
            },
        )

    def _get_token(self) -> str:
        try:
            return self._refresh_token()
        except requests.HTTPError:
            return self._authorise()

    # ---------------------------------------------------------------------
    # Semantic search & conversion
    # ---------------------------------------------------------------------

    def _semantic_search(self, query: str) -> list[dict]:
        """Raw semantic search – returns the provider’s JSON hits."""
        payload = {
            "query": query,
            "limit_count": self.k
        }
        return self._post("/api/v1/semantic/external/search", payload)

    def get_article(self, article_id: str, max_length: int = 0) -> str:
        article_info = self.get_article_info(article_id, max_length)
        return _get_article_text(self.base_url, article_info)

    def get_article_info(self, article_id: str, max_length: int = 0) -> str:
        payload = {
            "query": {
                "__filter": {
                            "id": article_id,
                            "editorContentAfterVersionAt": 1711433043
                        },
                "title": True,
                "space_id": True,
                "editorContentObject": {
                    "content": True
                }                
            }
        }
        return self._post("/api/v1/wiki/ql/article", payload)
    


    def _to_document(self, hit: dict) -> Document:
        """
        Convert one semantic-search hit (see sample payload below) into a
        LangChain Document.  We keep the full chunk text as page_content
        and put *everything else* into metadata so you can reference it
        later for filtering, citation, etc.

        Sample hit:
        {
            "space_id": "211ca458-...",
            "article_id": "f3d0cc75-...",
            "article_title": "Искусственный интеллект в медицине",
            "score": 0.80609,
            "text": "...",
            "chunk_token_length": 341,
            "offset": 0,
            "length": 1429
        }
        """
        return Document(
            page_content=f"{hit["text"]}\n\nСсылка на статью:https://kb.ileasing.ru/space/{hit["space_id"]}/article/{hit["article_id"]}",
            metadata={
                # provenance
                "space_id": hit["space_id"],
                "article_id": hit["article_id"],
                "article_title": hit["article_title"],
                # retrieval info
                "score": hit["score"],
                "chunk_token_length": hit["chunk_token_length"],
                "offset": hit["offset"],
                "length": hit["length"],
            },
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
        
    retriever = TeamlyRetriever("./auth.json", k=5)

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