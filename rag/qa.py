from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


prompt = ChatPromptTemplate.from_template("""
You are an academic research assistant.

Answer using ONLY the provided research paper context.

If the information is unavailable, say so.

Mention the paper name and page number
for supporting information.

Context:
{context}

Question:
{question}
""")


def ask_question(ensemble_retriever, question):

    retrieved_docs = ensemble_retriever.invoke(question)

    context = ""

    for doc in retrieved_docs:

        paper = doc.metadata.get("paper_name", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context += f"""
Paper: {paper}
Page: {page}
Content: {doc.page_content}
"""

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, retrieved_docs