from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente util que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historic_messages"),
    ("human", "Usuario: {question}")
])

historic_conversation = [
    HumanMessage(content="Usuario: Cual es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: Y cuantos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad."),
]

messages = chat_prompt.format_messages(
    historic_messages=historic_conversation,
    question="Puedes decirme algo interesante de su arquitectura?"
)

for message in messages:
    print(message.content)
