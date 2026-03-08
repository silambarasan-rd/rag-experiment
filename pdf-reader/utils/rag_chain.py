from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=75
)

def chunk_document(path="./silambarasan-resume.pdf"):
    pdf_content = PyPDFLoader(path)
    pages = pdf_content.load_and_split()

    chunks = splitter.split_documents(pages)

    return chunks 

def add_document(index, vector_store, path="./silambarasan-resume.pdf"):
    chunks = chunk_document(path)

    # To add new rows
    vector_store.add_documents(chunks)

if __name__ == "__main__":
    print("RAG CHAIN MODULE")
    print(chunk_document())