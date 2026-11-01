
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# 1. INITIALIZE LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ==========================================
# 2. CREATE PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are an academic research assistant.

Answer the user's question using ONLY the
provided research paper context.

Instructions:

1. Do not use outside knowledge.
2. If the information is unavailable, clearly say so.
3. Explain the answer in a clear and structured manner.
4. Mention the paper name and page number
   whenever you use supporting information.
5. Do not invent facts, citations or page numbers.
6. If different papers provide conflicting findings,
   explain the differences.

Context:
{context}

Question:
{question}

Answer:
""")


# ==========================================
# 3. ASK QUESTION USING HYBRID RETRIEVAL
# ==========================================

def ask_question(ensemble_retriever, question):

    # Retrieve relevant chunks using hybrid search
    retrieved_docs = ensemble_retriever.invoke(question)

    # Handle empty retrieval
    if not retrieved_docs:
        return (
            "I could not find relevant information "
            "in the uploaded research papers.",
            []
        )

    context = ""
    sources = []

    # ======================================
    # 4. BUILD CONTEXT AND SOURCE INFORMATION
    # ======================================

    seen_chunks = set()

    for doc in retrieved_docs:

        paper = doc.metadata.get(
            "paper_name", "Unknown"
        )

        # PyMuPDFLoader generally uses zero-based page indexes
        page = doc.metadata.get("page", None)

        if page is not None:
            page_number = int(page) + 1
        else:
            page_number = "Unknown"

        content = doc.page_content.strip()

        # Avoid duplicate chunks
        chunk_key = (paper, page_number, content)

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)

        context += f"""
Paper: {paper}
Page: {page_number}
Content: {content}

"""

        # Store source details
        sources.append({
            "paper": paper,
            "page": page_number,
            "content": content
        })

    # ======================================
    # 5. GENERATE ANSWER USING LLM
    # ======================================

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    # ======================================
    # 6. RETURN ANSWER AND SOURCES
    # ======================================

    return response.content, sources
