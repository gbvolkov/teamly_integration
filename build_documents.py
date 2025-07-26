import pandas as pd
from langchain_core.documents import Document  # LangChain Document class

def get_documents(df: pd.DataFrame) -> list[Document]:
    # Create a list of Documents from each row
    documents = []
    for _, row in df.iterrows():
        # Combine relevant fields into the page_content for retrieval
        content = (
            f"It System: {row['it_system']}\n"
            f"Problem Description: {row['problem_description']}\n"
            f"Problem Solution: {row['problem_solution']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "it_system": row['it_system'],
                "problem_description": row['problem_description'],
                "problem_solution": row['problem_solution']
            }
        )
        documents.append(doc)
    return documents
