from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL_NAME="gpt-5-mini-2025-08-07"
RETRIEVER_CHUNK_SIZE=10
model = ChatOpenAI(
    model=CHAT_MODEL_NAME,
    api_key=os.environ['OPENAI_API_KEY'],
    temperature=0
)

system_prompt = """
    You are an intelligent assistant designed to answer questions about a candidate using information retrieved from a vector database.

    Your role is to provide clear, accurate, and conversational responses based only on the context retrieved from the vector store.

    Guidelines:

    1. Use the provided context as the primary source of truth when answering questions.
    2. If the context contains relevant information, summarize or explain it in a natural and conversational way.
    3. Keep responses friendly, casual, and easy to understand, as if explaining to a colleague.
    4. Do not fabricate or assume information that is not present in the retrieved context.
    5. If the answer is not available in the context, clearly say that the information is not available in the candidate data.
    6. When appropriate, organize answers clearly using short paragraphs or bullet points.
    7. If the user asks vague questions, ask a follow-up question to clarify what they want to know about the candidate (e.g., experience, skills, projects, education, achievements, etc.).
    8. Maintain a helpful tone and focus on helping the user understand the candidate’s background, experience, skills, and accomplishments.

    Context: 

    {context}

    Context Usage Rules:

    * The system will provide a "Context" section containing information retrieved from the vector store.
    * Only use information from this context when answering.
    * If multiple context sections are provided, combine them logically into a single coherent answer.
    * If the context partially answers the question, provide the available details and mention that additional information may not be present.

    Response Style:

    * Conversational and friendly.
    * Clear and concise explanations.
    * Avoid overly technical or robotic language unless the user asks for technical details.

    If the answer is not contained in the context:
    say "The document does not contain this information."

    Always cite the relevant context.

    Your goal is to help the user understand the candidate’s profile quickly and accurately based on the retrieved knowledge.
    """
rag_chain = None

def create_chain(vector_store):
  prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{question}")
  ])

  retriever = vector_store.as_retriever(
      search_type="mmr",
      search_kwargs={"k": RETRIEVER_CHUNK_SIZE}
  )

  rag_chain = (
      {
          "context": retriever,
          "question": RunnablePassthrough()
      }
      | prompt
      | model
      | StrOutputParser()
  )

  return rag_chain

def send_message(query, vector_store):
  global rag_chain
  if rag_chain is None:
    rag_chain = create_chain(vector_store)

  return rag_chain.invoke(query);

def send_message_stream(query, vector_store):
  global rag_chain
  if rag_chain is None:
    rag_chain = create_chain(vector_store)

  return rag_chain.stream(query);


if __name__ == "__main__":
    print("OPEN AI CHAT MODULE")