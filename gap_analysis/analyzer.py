from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


gap_prompt = ChatPromptTemplate.from_template("""
You are an expert academic research analyst.

Analyze the following structured information
extracted from multiple research papers.

Identify potential research gaps in these categories:

1. Methodological gaps
2. Dataset gaps
3. Research scope gaps
4. Performance gaps
5. Unexplored combinations
6. Future research opportunities

Compare the papers rather than analyzing them individually.

For every potential gap, provide:

- Gap title
- Explanation
- Category
- Supporting paper names
- Evidence from the papers
- Suggested research direction

IMPORTANT:
- Do not invent findings.
- Every gap must have supporting evidence.
- Distinguish explicit limitations from gaps inferred
  through cross-paper comparison.
- Do not claim that a gap is globally unexplored.
- If evidence is insufficient, state that.

Paper analyses:
{paper_analyses}
""")


def find_research_gaps(paper_analyses):

    chain = gap_prompt | llm

    response = chain.invoke({
        "paper_analyses": paper_analyses
    })

    return response.content