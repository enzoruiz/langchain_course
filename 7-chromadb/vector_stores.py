from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

loader = DirectoryLoader(
    "/home/enzo/projects/langchain_course/7-chromadb/contratos_dummy",
    glob="**/*.pdf",
    loader_cls=PDFPlumberLoader,
)
documents = loader.load()

print(f"Se cargaron {len(documents)} documentos...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs_split = text_splitter.split_documents(documents)

print(f"Se crearon {len(docs_split)} chunks de texto...")

vectorstore = Chroma.from_documents(
    docs_split,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="/home/enzo/projects/langchain_course/7-chromadb/chroma_db"
)

query = "Cual es el inmueble que forma parte del contrato en el que participa María Jiménez Campos?"
print(f"Consulta: {query}")

results = vectorstore.similarity_search(
    query, k=3
)

print(f"Top 3 documentos mas similares a la consulta:")
for i, doc in enumerate(results):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
    print("====================================================================")
