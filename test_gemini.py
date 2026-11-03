from llm_factory import get_llm

llm = get_llm("gemini")

response = llm.invoke(
    "Explain Artificial Intelligence in simple words."
)

print(response.content)