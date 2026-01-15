from langchain_core.runnables import RunnableLambda

step1 = RunnableLambda(lambda x: f"Numero: {x}!")

def duplicate_text(text: str) -> str:
    return [text] * 2

step2 = RunnableLambda(duplicate_text)

chain = step1 | step2

response = chain.invoke(41)
print(response)
