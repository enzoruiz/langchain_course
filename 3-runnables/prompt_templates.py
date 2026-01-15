from langchain_core.prompts import PromptTemplate

template = """Eres un experto en marketing, sugiere un eslogan creativo para un producto: {product_name}.
"""

prompt = PromptTemplate(input_variables=["product_name"], template=template)

prompt_test = prompt.format(product_name="iPhone 14")
print(prompt_test)