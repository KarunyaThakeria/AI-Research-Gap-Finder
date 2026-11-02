
# =====================================================
# 1. LOAD ENVIRONMENT VARIABLES FIRST
# =====================================================

import os
import uuid
import json
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)

if not (os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")):
    raise ValueError(
        "OPENAI_API_KEY or API_KEY not found. Please check your .env file."
    )


# =====================================================
# 2. IMPORT LIBRARIES
# =====================================================

import streamlit as st

from document_processing.pdf_loader import load_papers
from document_processing.text_splitter import split_documents

from rag.vector_store import create_vector_store
from rag.qa import ask_question

from gap_analysis.extractor import extract_information
from gap_analysis.analyzer import find_research_gaps
from llm_factory import MODEL_OPTIONS


# =====================================================
# 3. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Research Gap Finder",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Gap Finder")

st.write(
    "Upload multiple research papers, ask questions "
    "and discover potential research gaps across models."
)


# =====================================================
# 3.1 SIDEBAR MODEL CONFIGURATION
# =====================================================

with st.sidebar:
    st.header("⚙️ Model Configuration")
    selected_model = st.selectbox(
        "Select LLM for Evaluation",
        options=MODEL_OPTIONS,
        index=0,
        help="Compare reasoning, latency, and tokens between models."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Active Model")
    st.info(f"**{selected_model}**")
    st.caption("Embeddings: `text-embedding-3-small` (OpenAI)")


# =====================================================
# 4. INITIALIZE SESSION STATE
# =====================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "ensemble_retriever" not in st.session_state:
    st.session_state.ensemble_retriever = None

if "paper_analyses" not in st.session_state:
    st.session_state.paper_analyses = None

if "processed_papers" not in st.session_state:
    st.session_state.processed_papers = []


# =====================================================
# 5. UPLOAD RESEARCH PAPERS
# =====================================================

uploaded_files = st.file_uploader(
    "Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)


# =====================================================
# 6. PROCESS RESEARCH PAPERS
# =====================================================

if st.button("Process Research Papers"):

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

    else:

        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)

        file_paths = []

        try:

            # -----------------------------------------
            # SAVE UPLOADED FILES
            # -----------------------------------------

            for file in uploaded_files:

                unique_name = str(uuid.uuid4()) + ".pdf"

                path = upload_dir / unique_name

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                file_paths.append(str(path))

            with st.spinner("Processing research papers..."):

                # -------------------------------------
                # PDF TEXT EXTRACTION
                # -------------------------------------

                documents = load_papers(file_paths)

                if not documents:
                    st.error(
                        "No readable text found in the uploaded PDFs."
                    )
                    st.stop()

                # -------------------------------------
                # TEXT CHUNKING
                # -------------------------------------

                chunks = split_documents(documents)

                if not chunks:
                    st.error("No text chunks were generated.")
                    st.stop()

                st.write(f"Extracted {len(documents)} pages.")
                st.write(f"Generated {len(chunks)} chunks.")

                # -------------------------------------
                # CREATE VECTOR STORE + HYBRID RETRIEVER
                # -------------------------------------

                vector_store, ensemble_retriever = create_vector_store(
                    chunks,
                    collection_name="papers_" + uuid.uuid4().hex
                )

                # Store both objects in session state
                st.session_state.vector_store = vector_store

                st.session_state.ensemble_retriever = (
                    ensemble_retriever
                )

                # -------------------------------------
                # STRUCTURED PAPER INFORMATION EXTRACTION
                # -------------------------------------

                analyses = []

                progress = st.progress(0)

                for index, path in enumerate(file_paths):

                    paper_name = os.path.basename(path)

                    # Get documents belonging to this paper
                    paper_docs = [
                        doc for doc in documents
                        if doc.metadata.get("paper_name") == paper_name
                    ]

                    paper_text = "\n".join(
                        doc.page_content for doc in paper_docs
                    )

                    analysis = extract_information(
                        paper_text[:25000],
                        model_name=selected_model
                    )

                    analyses.append({
                        "paper_name": paper_name,
                        "analysis": analysis.model_dump()
                    })

                    progress.progress(
                        (index + 1) / len(file_paths)
                    )

                st.session_state.paper_analyses = analyses

                st.session_state.processed_papers = file_paths

            st.success(
                "Research papers processed successfully!"
            )

            st.subheader("Processed Papers")

            for path in file_paths:
                st.write("📄 " + os.path.basename(path))

            st.info(
                f"Successfully processed {len(file_paths)} papers."
            )

        except Exception as e:

            st.error(
                f"Error while processing papers: {str(e)}"
            )


# =====================================================
# 7. RESEARCH PAPER Q&A AND GAP ANALYSIS
# =====================================================

if st.session_state.ensemble_retriever is not None:

    tab1, tab2 = st.tabs([
        "💬 Research Paper Q&A",
        "🔍 Research Gap Finder"
    ])

    # =================================================
    # TAB 1: RESEARCH PAPER Q&A
    # =================================================

    with tab1:

        st.subheader("Ask Questions Across Your Papers")

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What methodologies are used "
                "in these research papers?"
            )
        )

        if st.button("Get Answer"):

            if not question.strip():

                st.warning("Please enter a question.")

            else:

                try:

                    with st.spinner(
                        f"Searching with {selected_model}..."
                    ):

                        # Use hybrid retriever with selected model
                        answer, sources, metrics = ask_question(
                            st.session_state.ensemble_retriever,
                            question,
                            model_name=selected_model
                        )

                    st.markdown("### Answer")

                    st.write(answer)

                    # Model Evaluation Metrics Banner
                    st.markdown("#### ⚡ Performance & Token Metrics")
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("⏱️ Latency", f"{metrics['latency_sec']} s")
                    m_col2.metric("📥 Input Tokens", f"{metrics['input_tokens']:,}")
                    m_col3.metric("📤 Output Tokens", f"{metrics['output_tokens']:,}")
                    m_col4.metric("🚀 Throughput", f"{metrics['tokens_per_sec']} t/s")

                    st.markdown("### Supporting Sources")

                    if sources:

                        for source in sources:

                            with st.expander(
                                f"{source['paper']} | "
                                f"Page {source['page']}"
                            ):

                                st.write(source["content"])

                    else:

                        st.info(
                            "No supporting sources were returned."
                        )

                except Exception as e:

                    st.error(
                        f"Error while answering question: {str(e)}"
                    )


    # =================================================
    # TAB 2: RESEARCH GAP FINDER
    # =================================================

    with tab2:

        st.subheader("Discover Potential Research Gaps")

        st.write(
            "Compare the objectives, methodologies, datasets, "
            "findings, limitations and future work of "
            "your uploaded research papers."
        )

        if st.button("Analyze Research Gaps"):

            if not st.session_state.paper_analyses:

                st.warning(
                    "Please process research papers first."
                )

            else:

                try:

                    with st.spinner(
                        f"Comparing research papers using {selected_model}..."
                    ):

                        gap_results, metrics = find_research_gaps(
                            json.dumps(
                                st.session_state.paper_analyses
                            ),
                            model_name=selected_model
                        )

                    st.markdown("### Potential Research Gaps")

                    st.write(gap_results)

                    # Gap Analysis Metrics Banner
                    st.markdown("#### ⚡ Performance & Token Metrics")
                    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
                    g_col1.metric("⏱️ Latency", f"{metrics['latency_sec']} s")
                    g_col2.metric("📥 Input Tokens", f"{metrics['input_tokens']:,}")
                    g_col3.metric("📤 Output Tokens", f"{metrics['output_tokens']:,}")
                    g_col4.metric("🚀 Throughput", f"{metrics['tokens_per_sec']} t/s")

                except Exception as e:

                    st.error(
                        f"Error during gap analysis: {str(e)}"
                    )


# =====================================================
# 8. INITIAL MESSAGE
# =====================================================

else:

    st.info(
        "Upload and process research papers to get started."
    )
