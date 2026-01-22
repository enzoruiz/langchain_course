from langchain_community.document_loaders import PDFPlumberLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PDFPlumberLoader("/home/enzo/Descargas/documento.pdf")
pages = loader.load()

# dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
summaries = []

for chunk in chunks:
    response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}")
    summaries.append(response)

final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y commpleto: {" ".join(summaries)}")

print(final_summary.content)
