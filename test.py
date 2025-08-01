import requests
import json
from pprint import pprint

BASE_URL = "https://kb.ileasing.ru"

AUTH_DATA = "./auth2.json"
REDIRECT_URI = "https://kb.ileasing.ru/api/v1/semantic/external/search"

with open(AUTH_DATA, "r", encoding="utf-8") as f:
    auth_data = json.load(f)
client_id = auth_data.get("client_id")
client_secret = auth_data.get("client_secret")
auth_code = auth_data.get("auth_code")
redirect_uri = REDIRECT_URI

def authorize_integration():
    """
    Calls the POST api/v1/auth/integration/authorize endpoint to obtain access and refresh tokens.

    :param base_url: Base URL of the API (e.g., "https://your-api-domain.com")
    :param client_id: Integration client identifier
    :param redirect_uri: Redirect URI registered for your integration
    :param client_secret: Integration secret key
    :param code: Authorization code obtained from the OAuth flow
    :return: Parsed JSON response (dict) if successful; raises an exception otherwise
    """
    url = f"{BASE_URL}/api/v1/auth/integration/authorize"
    payload = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "client_secret": client_secret,
        "code": auth_code
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an HTTPError if the response was not 2xx

    return response.json()

def refresh_auth(refresh_token):
    """
    Calls the POSTapi/v1/auth/integration/refresh endpoint to refresh tokens.

    :param base_url: Base URL of the API (e.g., "https://your-api-domain.com")
    :param client_id: Integration client identifier
    :param redirect_uri: Redirect URI registered for your integration
    :param client_secret: Integration secret key
    :param code: Authorization code obtained from the OAuth flow
    :return: Parsed JSON response (dict) if successful; raises an exception otherwise
    """
    url = f"{BASE_URL}/api/v1/auth/integration/refresh"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an HTTPError if the response was not 2xx

    return response.json()

def auth_post(url: str, payload: dict) -> str:
    # No bearer header during auth flow
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Accept": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    refresh_token = data.get("refresh_token")
    access_token  = data.get("access_token")

    # Persist fresh refresh-token
    auth_code = refresh_token or access_token
    _auth_data["auth_code"] = auth_code
    with open(self._auth_path, "w", encoding="utf-8") as fp:
        json.dump(self._auth_data, fp, ensure_ascii=False, indent=4)

    return access_token

def refresh_token(client_id, client_secret, redirect_uri, auth_code) -> str:
    return self._auth_post(
        f"{self.base_url}/api/v1/auth/integration/refresh",
        {
            "client_id":     client_id,
            "client_secret": client_secret,
            "refresh_token": auth_code,
        },
    )

def authorise(client_id, client_secret, redirect_uri, auth_code) -> str:
    return self._auth_post(
        f"{self.base_url}/api/v1/auth/integration/authorize",
        {
            "client_id":     client_id,
            "client_secret": client_secret,
            "redirect_uri":  redirect_uri,
            "code":          auth_code,
        },
    )

def get_token(client_id, client_secret, redirect_uri, auth_code) -> str:
    try:
        return refresh_token(client_id, client_secret, redirect_uri, auth_code)
    except requests.HTTPError:
        return authorise(client_id, client_secret, redirect_uri, auth_code)

def get_user(access_token, user_name):
    url = f"{BASE_URL}/api/v1/ql/account-users"
    headers = {
        "X-Account-Slug": "default",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    payload = {
        "query": {
            "__filter": {
                "__text": {
                    "query": user_name
                },
                "types": ['employee'],
            },
            "__sort": [
                {
                    "createdAt": "desc"
                }
            ],
            "__pagination": {
                "page": 1,
                "perPage": 20
            },
            "id": True,
            "userId": True,
            "extUserId": True,
            "name": True,
            "surname": True,
            "active": True,
            "userType": True,
            "userRole": True,
            "user": {
                "email": True,
                "avatar": {
                    "path": True
                }
            }
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an HTTPError if the response was not 2xx

    return response.json()

def semantic_search(access_token, user_id, query):
    url = f"{BASE_URL}/api/v1/semantic/external/search"

    headers = {
        "X-Account-Slug": "default",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    payload = {
        "user_id": user_id,
        "limit": 500,
        "check_permissions": False,
        "query": query
    }    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an HTTPError if the response was not 2xx

    return response.json()

def list_spaces(access_token):
    url = f"{BASE_URL}/api/v1/wiki/ql/spaces"

    headers = {
        "X-Account-Slug": "default",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    payload = {
        "query": {
            "__filter": {
                "keeping_types": [
                    "default"
                ],
                "__text": {},
                "__nested": {
                    "__text": {
                        "query": ""
                    }
                }
            },
            "__sort": [
                {
                    "pinned_at": "desc"
                },
                {
                    "created_at": "desc"
                }
            ],
            "__pagination": {
                "page": 1,
                "per_page": 50
            },
            "id": True,
            "title": True,
            "description": True,
            "main_article": {
                "id": True,
                "icon_color": True,
                "image": {
                    "cover_id": True
                }
            }
        }
    }    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an HTTPError if the response was not 2xx

    return response.json()



if __name__ == "__main__":
    # Example usage:


    try:
        with open(AUTH_DATA, "r", encoding="utf-8") as f:
            auth_data = json.load(f)
        client_id = auth_data.get("client_id")
        client_secret = auth_data.get("client_secret")
        auth_code = auth_data.get("auth_code")

        #auth_response = authorize_integration()

        auth_response = refresh_auth(auth_code)

        # Example of how to extract values from the response:
        #user_info = auth_response.get("user", {})
        access_token = auth_response.get("access_token")
        #refresh_token = auth_response.get("refresh_token")
        #accounts = auth_response.get("accounts", [])
        #account_users = auth_response.get("account_users", {})

        #print("User ID:        ", user_info.get("id"))
        #print("User Full Name: ", user_info.get("fullName"))
        #print("Email:          ", user_info.get("email"))
        #print("Access Token:   ", access_token)
        #print("Refresh Token:  ", refresh_token)
        #print("Accounts:       ", accounts)
        #print("Account Users:  ", account_users)


        auth_data["auth_code"] = auth_response.get("refresh_token")
        with open(AUTH_DATA, "w", encoding="utf-8") as f:
            auth_data = json.dump(auth_data, f, ensure_ascii=False, indent=4)

        users = get_user(access_token, "volkovgb@ileasing.ru")
        #pprint(users)
        user_id = users.get("items")[0].get("userId")


        headers = {
            "X-Account-Slug": "default",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        # Ссылка на получение метаданных файла
        #metadata_url = "https://kb.ileasing.ru/api/v1/disk/storage/3eeb16bd-6cc1-4283-8508-78ed16447ddb/doc/853ba3d8-1aba-4eea-b5fa-5c6f09284182"

        # Ссылка для скачивания самого файла
        #download_url = metadata_url + "/view"


        #response = requests.get(metadata_url, headers=headers)
        #response.raise_for_status()
        #data = response.json()
        #filename = data.get("title", "downloaded_file")

        # Загружаем файл
        #file_response = requests.get(download_url, headers=headers)
        #if file_response.status_code == 200:
        #    with open(filename, "wb") as f:
        #        f.write(file_response.content)
        #    print(f"Файл '{filename}' успешно загружен и сохранён.")
        #else:
        #    print(f"Ошибка при загрузке файла: {file_response.status_code}")


        #spaces = list_spaces(access_token)
        #pprint(spaces)

        #access_token = auth_code
        found = semantic_search(access_token, user_id, "настройка микросип")
        pprint(found)

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
    except Exception as err:
        print(f"An unexpected error occurred: {err}")