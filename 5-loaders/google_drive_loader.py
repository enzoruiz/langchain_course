# from langchain_community.document_loaders import GoogleDriveLoader # OLD
from langchain_google_community import GoogleDriveLoader

credentials_path = "credentials.json"
token_path = "token.json"
folder_id = "123"

loader = GoogleDriveLoader(
    folder_id=folder_id,
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True,
    file_types=["pdf", "document"]
)

documents = loader.load()
print(documents)
