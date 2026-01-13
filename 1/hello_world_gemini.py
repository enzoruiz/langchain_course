from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

question = "En que año llego el ser humano a la luna por primera vez?"
print("Pregunta: ", question)

answer = llm.invoke(question)
print("Respuesta del modelo: ", answer.content)
