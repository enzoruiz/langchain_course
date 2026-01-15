import json
from langchain_core.runnables import RunnableLambda
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

def merge_results(data: dict) -> dict:
    return {
        "summary": data["summary"],
        "sentiment": data["sentiment_data"]["sentiment"],
        "reason": data["sentiment_data"]["reason"],
    }

def process_one(text: str) -> str:
    summary = generate_summary(text)
    sentiment_data = analyze_sentiment(text)
    return merge_results({"summary": summary, "sentiment_data": sentiment_data})

process = RunnableLambda(process_one)

chain = preprocessor | process

# DATA DE PRUEBA
textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]
 
for texto in textos_prueba:
    resultado = chain.invoke(texto)
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-" * 50)
