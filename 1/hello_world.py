from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

question = "En que año llego el ser humano a la luna por primera vez?"
print("Pregunta: ", question)

answer = llm.invoke(question)
print("Respuesta del modelo: ", answer.content)
