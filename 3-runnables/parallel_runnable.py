import json
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def preprocess_test(text: str) -> str:
    text = text.strip()[:500]
    return text

preprocessor = RunnableLambda(preprocess_test)

def generate_summary(text: str) -> str:
    propmt = f"Resume en una sola oración el siguiente texto: {text}"
    response = llm.invoke(propmt)
    return response.content

runnable_summary = RunnableLambda(generate_summary)

def analyze_sentiment(text: str) -> dict:
    prompt = f"""Indica la sentimiento de este texto.
    Responde UNICAMENTE en formato JSON válido:
    {{"sentiment": "positivo|negativo|neutro", "reason": "justificación breve"}}

    Texto: {text}
    """

    response = llm.invoke(prompt)
    try:
        response = json.loads(response.content)
    except json.JSONDecodeError:
        response = {"sentiment": "neutro", "reason": "Error en analisis."}

    return response

runnable_analyze_sentiment = RunnableLambda(analyze_sentiment)

def merge_results(data: dict) -> dict:
    return {
        "summary": data["summary"],
        "sentiment": data["sentiment_data"]["sentiment"],
        "reason": data["sentiment_data"]["reason"],
    }

runnable_merge_results = RunnableLambda(merge_results)

parallel_analysis = RunnableParallel({
    "summary": runnable_summary,
    "sentiment_data": runnable_analyze_sentiment,
})

chain = preprocessor | parallel_analysis | runnable_merge_results

# DATA DE PRUEBA
reviews_batch = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]

response_batch = chain.batch_invoke(reviews_batch)

print(response_batch)
