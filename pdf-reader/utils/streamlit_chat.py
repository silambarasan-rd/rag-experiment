import streamlit as st
from utils.pinecone_db import init_pinecone_db
from utils.rag_chain import add_document
from utils.openai_chat import send_message_stream
import os

# Initializing the DB
index, vector_store = init_pinecone_db()

UPLOAD_FOLDER = "documents"

# create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def add_pdf_document(path):
    # Adding new document to the Pinecone Index (for specific namespace)
    add_document(index, vector_store, path)

def chat_model(query):
    response = send_message_stream(query, vector_store)
    return response

def init_ui():
  st.set_page_config(page_title="Resume Analyzer")
  st.title("Resume Analyzer")

  if "messages" not in st.session_state:
    st.session_state['messages'] = []

  for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  uploaded_file = st.file_uploader(
    "Upload a Resume (only PDF)",
    type=["pdf"]
  )

  if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(f"documents/{uploaded_file.name.lower()}", "wb") as f:
      f.write(uploaded_file.getbuffer())
    
    st.success(f"Uploaded: {file_path}")

    add_pdf_document(file_path)
    

  prompt = st.chat_input("Ask me about the documents you've uploaded!")

  if prompt:
    st.session_state['messages'].append({
      "role": "user",
      "content": prompt
    })

    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      placeholder = st.empty()

      placeholder.markdown("⏳ Thinking...")
      stream_response = chat_model(prompt)

      full_text = ""
      for chunk in stream_response:
        full_text += chunk
        placeholder.markdown(full_text + "▌")

      placeholder.markdown(full_text)

      st.session_state['messages'].append({
        "role": "assistant",
        "content": full_text
      })

if __name__ == "__main__":
  print("STREAMLIT CHAT MODULE")
  init_ui()