from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader, WebBaseLoader

# LOAD A PDF
loader = PDFPlumberLoader("/home/enzo/Descargas/Documento.pdf")
pages = loader.load()

for i, page in enumerate(pages):
    print(f"Pagina: {i}")
    print(f"Contenido: {page.page_content}")
    print(f"Metadatos: {page.metadata}")
    print("========================================")


# loader_web = WebBaseLoader("https://techmind.ac")
# data = loader_web.load()
# print(data)
