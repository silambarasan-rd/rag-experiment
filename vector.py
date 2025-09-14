from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("ev_final.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for index, row in df.iterrows():
        document = Document(
            page_content=row["address"] + " " + row["city"] + " " + row["country"] + " " + row["name"] + " " + row["vendor_name"],
            metadata={"uid": row["uid"], "name": row["name"], "address": row["address"], "city": row["city"], "country": row["country"], "vendor_name": row["vendor_name"], "logo_url": row["logo_url"]},
            id=str(index)
        )

        ids.append(str(index))
        documents.append(document)

vector_store = Chroma(
    collection_name="ev_stations",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)
