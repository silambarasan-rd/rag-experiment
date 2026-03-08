import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME="rag-experiment-py"
EMBEDDING_DIM = 1536
# only for testing, in a production application this will be dynamic as for each user
PINECONE_NAMESPACE_NAME = "silambarasan-r-7708863236"
EMBEDDING_MODEL_NAME="text-embedding-3-small"

pinecone_db = Pinecone(
    api_key=os.environ['PINECONE_API_KEY']
)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME
)

def init_pinecone_db():
    existing_indexes = pinecone_db.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        pinecone_db.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=spec
        )

    index = pinecone_db.Index(INDEX_NAME)

    vector_store = PineconeVectorStore(
        embedding=embedding_model,
        index=index,
        namespace=PINECONE_NAMESPACE_NAME
    )

    return index, vector_store


if __name__ == "__main__":
    print("PINECONE DB MODULE")
    index, store = init_pinecone_db()

    print("Checking the Index Stats: ")
    print(index.describe_index_stats())