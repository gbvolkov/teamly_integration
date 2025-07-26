from teamly_retriever import TeamlyRetriever
import json
from load_table import get_data_from_json
from build_index import get_retrievers

tr = TeamlyRetriever("./auth.json", k=40)

article_info = tr.get_article_info("3fdb4f97-2246-4b9e-b477-e9d7d8a2eb86")
raw_doc = article_info["editorContentObject"]["content"]
doc = json.loads(raw_doc)

query = "не работает бронирование"

df = get_data_from_json(doc)

(idx_vectors, idx_bm25) = get_retrievers(df)
v_res = idx_vectors.similarity_search(query, k=5)
bm25_res = idx_bm25.get_relevant_documents(query)[:5]

print(v_res)
print(bm25_res)