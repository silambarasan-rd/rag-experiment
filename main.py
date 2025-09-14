from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about EV Charging Stations in India.

Here are the list of EV Charging Stations: {charging_stations}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
   print("\n\n")
   print("======================================")
   question = input("Ask your query (or q to quit): ")
   print("--------------------------------------")
   print("\n\n")

   if question == "q":
       break
   
   documents = retriever.invoke(question)
   result = chain.invoke({ "charging_stations": documents, "question": question})
   print(result)
