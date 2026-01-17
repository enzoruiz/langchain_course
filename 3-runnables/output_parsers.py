from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class AnalysisText(BaseModel):
    summary: str = Field(description="Resumen breve del texto")
    sentiment: str = Field(description="Sentimiento del texto (Positivo, neutro o negativo)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

structured_llm = llm.with_structured_output(AnalysisText)

test_data = "me encanto la nueva pelicula de accion, tiene muchos efectos especiales y emocion."

response = structured_llm.invoke(f"Analiza el siguiente texto: {test_data}")

print(response)
