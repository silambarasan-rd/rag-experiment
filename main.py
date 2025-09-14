from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an advanced AI assistant specializing in Electric Vehicle (EV) Charging Stations in India.

You have access to a limited sample dataset of EV Charging Stations in India (not a complete list). Use this data to answer questions, and clearly indicate if the information is based on the available sample. If a query cannot be answered with the provided data, explain the limitation and suggest general advice or next steps.

For each question, use chain-of-thought reasoning to analyze and answer:
1. Analyze the user's question and clarify its intent.
2. Review the provided sample data for relevant information.
3. Reason step-by-step to connect the data to the user's query.
4. If the answer is not directly available, explain the limitation and provide helpful suggestions.
5. Present the final answer in a clear, concise, and professional manner. Include practical tips or additional context if relevant.

Important: Do not simply output the chain-of-thought steps as the final answer. Always provide a direct answer or helpful suggestion at the end, summarizing your reasoning. If the data does not contain the requested information, clearly state this and offer alternative ways to find the answer.

Here is the available sample data on EV Charging Stations: {charging_stations}

User question: {question}
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
