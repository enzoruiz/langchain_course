from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

template = PromptTemplate(
    input_variables=["name"],
    template="Saluda al usuario con su nombre.\nNombre del usuario: {name}\nAsistente:",
)

chain = template | chat

response = chain.invoke({"name": "Bram"})
print(response.content)
