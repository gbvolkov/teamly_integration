from teamly_retriever import TeamlyRetriever
import json
from load_table import get_data_from_json
from build_index import get_retrievers
import pandas as pd


tr = TeamlyRetriever("./auth.json", k=40)


#article_info = tr.get_article_info("e7a19a56-d067-4023-b259-94284ec4e16b")


query = "не работает бронирование"
#tr.load_sd_articles_index()
docs = tr.invoke(query)
print(docs)

articles = [
    "e7a19a56-d067-4023-b259-94284ec4e16b", 
    "a1038bbc-e5d9-4b5a-9482-2739c19cb6cb",
    "dd64ab73-50ea-4d48-83f0-8dcef88512cb",
    "25eef990-b807-4b13-90e7-68ecadfe7a57",
    "e834824e-b9bc-42fb-b9a2-95d2b1b3a125",
    "3fdb4f97-2246-4b9e-b477-e9d7d8a2eb86",
]

df = pd.DataFrame()
for article_id in articles:
    article_info = tr.get_article_info(article_id)
    space_id = article_info["space_id"]
    article_title = article_info["title"]

    raw_doc = article_info["editorContentObject"]["content"]
    doc = json.loads(raw_doc)
    arcticle_df = get_data_from_json(doc, space_id, article_id, article_title)
    df = pd.concat([df, arcticle_df], ignore_index=True)

(idx_vectors, idx_bm25) = get_retrievers(df)

query = "не работает бронирование"
v_res = idx_vectors.similarity_search(query, k=5)
bm25_res = idx_bm25.invoke(query)[:5]

print(v_res)
print(bm25_res)