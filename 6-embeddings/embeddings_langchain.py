from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

text_1 = "La capital de Francia es París."
text_2 = "Paris es un nombre comun para perros."

vector_1 = embeddings.embed_query(text_1)
vector_2 = embeddings.embed_query(text_2)

print(f"Dimension de los vectores: {len(vector_1)}")

cos_sim = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

print(f"Similitud coseno entre vector 1 y 2: {cos_sim:.3f}")
