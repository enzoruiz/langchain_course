from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/enzo/projects/langchain_course/chroma_db"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

query = "Donde se encuentra el local del contrato en el que participa María Jiménez Campos?"
print(f"Consulta: {query}")

results = retriever.invoke(query)

print(f"Top 2 documentos mas similares a la consulta:")
for i, doc in enumerate(results):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
    print("====================================================================")
