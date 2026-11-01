from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class PaperAnalysis(BaseModel):

    title: str

    objectives: List[str]

    methodologies: List[str]

    datasets: List[str]

    findings: List[str]

    limitations: List[str]

    future_work: List[str]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

structured_llm = llm.with_structured_output(
    PaperAnalysis
)


extraction_prompt = ChatPromptTemplate.from_template("""
You are an academic research paper analyst.

Analyze the following research paper.

Extract:
1. Research objectives
2. Methodologies and algorithms
3. Datasets used
4. Main findings
5. Limitations
6. Future research directions

Only extract information supported by the paper.
Do not invent missing information.

If information is unavailable, return an empty list.

Paper content:
{paper_text}
""")


def extract_information(paper_text):

    chain = extraction_prompt | structured_llm

    result = chain.invoke({
        "paper_text": paper_text
    })

    return result