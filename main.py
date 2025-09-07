from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a Thirukural.

Here are the list of thirukurals: {thirukurals}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
   print("\n\n")
   print("======================================")
   question = input("Ask your question (or q to quit): ")
   print("--------------------------------------")
   print("\n\n")

   if question == "q":
       break
   
   documents = retriever.invoke(question)
   result = chain.invoke({ "thirukurals": documents, "question": question})
   print(result)
