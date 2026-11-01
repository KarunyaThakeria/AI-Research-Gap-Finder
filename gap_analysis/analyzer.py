import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ----------------------------------------------------------------------
# Pass 1: draft gaps, strictly from the supplied text
# ----------------------------------------------------------------------
gap_prompt = ChatPromptTemplate.from_template("""
You are a careful research analyst. You will be given structured information extracted from several research papers.
Your job is to report research gaps that are SUPPORTED BY THE TEXT BELOW. Accuracy matters far more than quantity.

SOURCE RULES (mandatory)
1. Use ONLY the text under INPUT. Do not use outside knowledge, other papers, or what you think is true about the field.
2. Never invent findings, datasets, numbers, limitations, or quotations.
3. Every gap must be backed by evidence copied EXACTLY from the INPUT (same words, in quotation marks). If you cannot copy a supporting sentence or phrase, do not report the gap.
4. If you paraphrase, do not use quotation marks.
5. Do NOT claim a topic is "unexplored", "missing" or "never studied". You only know about the papers given. You may say "not addressed in the provided papers", and only as an Inferred gap.
6. A paper not mentioning something is NOT evidence of a gap by itself.
7. If a paper has no limitation or future-work information in the INPUT, say so; do not guess one.

GAP TYPES
- Explicit: the authors themselves state this limitation or future work in the INPUT. You must quote it.
- Inferred: you compare two or more papers in the INPUT and found a difference, conflict, or limitation that follows from quoted evidence in EACH paper you cite. An Inferred gap needs quotes from at least two different papers.
Do not use any other type. Do not propose method combinations unless both parts are quoted from the INPUT.

HOW MANY GAPS
There is no target number. Return 0, 1, 2 or more gaps, depending only on the evidence. Returning "No sufficiently supported gaps" is a correct answer when evidence is thin. Merge gaps that describe the same problem.

OUTPUT FORMAT (use exactly this, one block per gap)

### G1: <specific title>
- **Type:** Explicit or Inferred
- **Papers:** <paper names>
- **Evidence:** "<exact quote>" (<paper name>, <page or section if given in the INPUT, otherwise omit>)
- **Why this is a gap:** <1-2 sentences that follow directly from the evidence>
- **Suggested direction:** <1 sentence, must follow directly from the evidence>
- **Confidence:** High (author states it) / Medium (clear cross-paper comparison) / Low (weak support)

After all gaps add:

### Papers with no usable limitation or future-work information
<list paper names, or write "None">

### Summary
<number of gaps, how many Explicit vs Inferred, and one sentence on any weakness in the available information>

INPUT
{paper_analyses}
""")


# ----------------------------------------------------------------------
# Pass 2: fact-check the draft against the same source
# ----------------------------------------------------------------------
verify_prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checker. Below are an INPUT (the only source of truth) and a DRAFT list of research gaps written from it.
Check the DRAFT against the INPUT and return a corrected version.

For every gap in the DRAFT:
1. Check each quotation. A quote is valid only if those words really appear in the INPUT. If a quote is not found, remove it. If no valid evidence remains, delete the gap.
2. If a gap is labelled Explicit, confirm the INPUT shows the authors themselves stating that limitation or future work. If not, change it to Inferred, or delete it if it is not supported by quotes from at least two papers.
3. Delete any gap that claims something is "unexplored", "never studied" or "missing in the field". Rewrite it as "not addressed in the provided papers" only if it still has quoted support.
4. Delete any gap that relies only on a paper not mentioning something.
5. Check that numbers, dataset names and method names in the gap appear in the INPUT. Fix or delete if not.
6. Make sure "Why this is a gap" and "Suggested direction" follow from the quoted evidence and add no new facts.
7. Lower the confidence if the support is weaker than stated.

These quotes were automatically found NOT to appear in the INPUT. Remove them:
{unverified_quotes}

Keep the same output format as the DRAFT. Renumber the gaps (G1, G2, ...). Update the Summary counts.
If no gap survives, output exactly: "No sufficiently supported research gaps could be identified from the provided papers."
At the end add a short section "### Removed during verification" listing each removed gap title with a reason of at most 10 words (write "None" if nothing was removed).
Do not add new gaps.

INPUT
{paper_analyses}

DRAFT
{draft}
""")


# ----------------------------------------------------------------------
# Programmatic quote check (no LLM, so it cannot hallucinate)
# ----------------------------------------------------------------------
def _normalize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _find_unverified_quotes(draft, source):
    """Return quoted passages in the draft that do not appear in the source text."""
    source_norm = _normalize(source)
    quotes = re.findall(r'["\u201c]([^"\u201d]{15,}?)["\u201d]', draft)
    return [q for q in quotes if _normalize(q) not in source_norm]


def find_research_gaps(paper_analyses):
    # Pass 1: draft
    draft = (gap_prompt | llm).invoke({
        "paper_analyses": paper_analyses
    }).content

    # Check quotes in code
    unverified = _find_unverified_quotes(draft, paper_analyses)
    unverified_text = "\n".join(f"- {q}" for q in unverified) if unverified else "None"

    # Pass 2: verification
    final = (verify_prompt | llm).invoke({
        "paper_analyses": paper_analyses,
        "draft": draft,
        "unverified_quotes": unverified_text
    }).content

    return final