from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("thirukural_explanation.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for index, row in df.iterrows():
        document = Document(
            page_content=row["Chapter_Name"] + " " + row["Explanation"] + " " + row["Translation"],
            metadata={"kural": row["Verse"], "chapter": row["Chapter_Name"]},
            id=str(index)
        )

        ids.append(str(index))
        documents.append(document)

vector_store = Chroma(
    collection_name="thirukural_explanations",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    # vector_store.persist()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
