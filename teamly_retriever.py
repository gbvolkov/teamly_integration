from __future__ import annotations

import functools
import json
import requests
from typing import List, Optional

from langchain.schema import Document
from langchain.schema.runnable import RunnableConfig
from langchain.schema.retriever import BaseRetriever

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

    # --------------------------------- public LangChain hook -------------

    def _get_relevant_documents(        # type: ignore[override]
        self,
        query: str,
        *,
        config: Optional[RunnableConfig] = None,  # for LC internal plumbing
    ) -> List[Document]:
        """Synchronous retrieval used by most LC components."""
        raw_hits = self._semantic_search(query)[: self.k]
        return [self._to_document(hit) for hit in raw_hits]

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
            "limit": 2048
        }
        return self._post("/api/v1/semantic/external/search", {"query": query})

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

    llm = ChatOpenAI(model="gpt-4.1")
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

    result = rag_chain.invoke({"input": "КАкие субсидии даёт МПТ?"})
    pprint(result)