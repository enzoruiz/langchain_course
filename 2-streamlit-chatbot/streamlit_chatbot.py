from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st

# Configurar la pagina de la app
st.set_page_config(
    page_title="Chatbot con LangChain",
    page_icon="🤖",
)
st.title("Chatbot con LangChain")
st.markdown("Un bot de ejemplo construido con LangChain y Streamlit")

with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"])

    chat_model = ChatOpenAI(model=model_name, temperature=temperature)

prompt_template = PromptTemplate(
    input_variables=["history", "question"],
    template="""Eres un asistente útil y amigable llamado BramBot.
    
    Historial de la conversación:
    {history}

    Responde de manera clara y concisa a la siguiente pregunta:
    {question}""",
)
chain = prompt_template | chat_model

# Inicializar el historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial de mensajes en la interfaz
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "assistant" if isinstance(message, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(message.content)

if st.button("🗑️ Nueva conversación"):
    st.session_state.messages = []
    st.rerun()

# Mostrar el input de la interfaz
question = st.chat_input("Pregunta")

if question:
    # Mostrar el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(question)

    # Mostrar la respuesta del modelo en la interfaz
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Streaming de respuesta del modelo
            for chunk in chain.stream({"history": st.session_state.messages, "question": question}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        # Guardamos los mensajes del usuario y el modelo en la memoria de streamlit
        st.session_state.messages.append(HumanMessage(content=question))
        st.session_state.messages.append(AIMessage(content=full_response))
    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")

