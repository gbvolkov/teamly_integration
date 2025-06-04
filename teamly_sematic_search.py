import json
import functools
import requests
from langchain.schema.retriever import BaseRetriever

def authorization_wrapper(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except requests.HTTPError as http_err:
            resp = http_err.response
            status = resp.status_code
            if status != 401:
                raise
            self.auth_code = self.get_token()
            return func(self, *args, **kwargs)
    return wrapper


class TeamlyRetriever():
    def __init__(self, auth_data_store: str):
        with open(auth_data_store, "r", encoding="utf-8") as f:
            auth_data = json.load(f)
        self.auth_data_store = auth_data_store
        self.auth_data = auth_data

        self.base_url = auth_data.get("base_url")
        self.client_id = auth_data.get("client_id")
        self.client_secret = auth_data.get("client_secret")
        self.auth_code = auth_data.get("auth_code")
        self.redirect_uri = auth_data.get("redirect_uri")
        
        self.headers = {
            "X-Account-Slug": "default",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_code}"
        }

    @authorization_wrapper
    def _post(self, url, payload):
        headers = {
            "X-Account-Slug": "default",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_code}"
        }
        response = requests.post(f"{self.base_url}{url}", json=payload, headers=headers)
        
        response.raise_for_status()  # Raise an HTTPError if the response was not 2xx

        return response.json()

    def _auth_post(self, url, headers, payload):
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an HTTPError if the response was not 2xx
        resp_data = response.json()
        refresh_token =  resp_data.get("refresh_token")
        access_token =  resp_data.get("access_token")
        self.auth_code = access_token
        self.auth_data["auth_code"] = refresh_token
        with open(self.auth_data_store, "w", encoding="utf-8") as f:
            json.dump(self.auth_data, f, ensure_ascii=False, indent=4)
        return access_token        

    def _refresh_token(self):
        url = f"{self.base_url}/api/v1/auth/integration/refresh"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.auth_code
        }

        return self._auth_post(url, headers, payload)

    def _authorise(self):
        url = f"{self.base_url}/api/v1/auth/integration/authorize"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "client_secret": self.client_secret,
            "code": self.auth_code
        }
        return self._auth_post(url, headers, payload)

    def get_token(self):
        try:
            token = self._refresh_token()
        except:
            token = self._authorise()
        return token

    def semantic_search(self, query):
        url = f"/api/v1/semantic/external/search"

        payload = {
            "query": query
        }
        return self._post(url, payload)  

if __name__ == "__main__":
    from pprint import pprint
    AUTH_DATA = "./auth.json"

    retriever = TeamlyRetriever(AUTH_DATA)
    try:
        #pages = retriever.semantic_search("Какова цель создания реестра расчетов СВО по изъятым договорам лизинга и какую информацию он должен содержать?")
        pages = retriever.semantic_search("кей юзеры")
        pprint(pages)
    except requests.HTTPError as http_err:
        resp = http_err.response
        try:
            err_json = resp.json()
            print(f"HTTP error occurred: {http_err} – Response body: ")
            
            for field, messages in err_json.items():

                if isinstance(messages, str):
                    msg = messages
                else:
                    if messages:
                        msg = ", ".join(messages)
                    else:
                        msg = "None"
                print(f"    {field}: {msg}")
        except:
            print(f"HTTP error occurred: {http_err} – Response body: {resp.text}")
