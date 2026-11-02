
import time
from langchain_core.prompts import ChatPromptTemplate
from llm_factory import get_llm, extract_metrics


# ==========================================
# 2. CREATE PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are an expert academic research assistant specializing in
analyzing and synthesizing information from research papers.

Your task is to answer the user's question accurately and
completely using ONLY the provided research paper context below.

## Instructions

### Grounding Rules
1. Use ONLY the information present in the provided context.
2. Do not use outside knowledge, assumptions, or general facts.
3. Do not invent facts, findings, citations, or page numbers.
4. If the context does not contain sufficient information to
   answer the question, clearly state:
   "The uploaded papers do not contain sufficient information
   to answer this question."

### Citation Rules
5. Every claim must be supported by a citation in this format:
   (Paper: <paper_name>, Page: <page_number>)
6. If multiple papers support the same point, cite all of them.
7. Do not cite a paper for information it did not provide.

### Answer Quality Rules
8. Synthesize information across all relevant papers into a
   single coherent answer — do not answer per paper separately
   unless the question explicitly asks for a per-paper breakdown.
9. If papers present conflicting or differing findings on the
   same topic, explicitly highlight the differences and explain
   what each paper found.
10. If the context only partially answers the question, provide
    what is available and clearly state what is missing.
11. Avoid vague, generic, or filler statements.
    Every sentence must add specific value to the answer.

### Structure Rules
12. Structure your answer clearly using the format below.

## Answer Format

**Direct Answer:**
A concise, direct answer to the question in 1-3 sentences.

**Detailed Explanation:**
A thorough explanation with specific details, evidence, and
in-text citations from the provided context.

**Key Points:**
- Bullet point summary of the most important findings
  relevant to the question.

**Conflicting Findings (if any):**
If papers disagree, clearly explain the differences here.
Omit this section if there are no conflicts.

## Context

{context}

## Question

{question}

## Answer
""")


# ==========================================
# 3. ASK QUESTION USING HYBRID RETRIEVAL
# ==========================================

def ask_question(ensemble_retriever, question, model_name: str = "OpenAI (GPT-4o-mini)"):

    # Retrieve relevant chunks using hybrid search
    retrieved_docs = ensemble_retriever.invoke(question)

    # Handle empty retrieval
    if not retrieved_docs:
        return (
            "I could not find relevant information "
            "in the uploaded research papers.",
            [],
            {"latency_sec": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "tokens_per_sec": 0}
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
    # 5. GENERATE ANSWER USING SELECTED LLM
    # ======================================

    selected_llm = get_llm(model_name)
    chain = prompt | selected_llm

    start_time = time.time()
    response = chain.invoke({
        "context": context,
        "question": question
    })
    metrics = extract_metrics(response, start_time)

    # ======================================
    # 6. RETURN ANSWER, SOURCES, AND METRICS
    # ======================================

    return response.content, sources, metrics
