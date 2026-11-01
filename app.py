import streamlit as st
import os
import uuid
import json

from document_processing.pdf_loader import load_papers
from document_processing.text_splitter import split_documents
from rag.vector_store import create_vector_store
from rag.qa import ask_question

from gap_analysis.extractor import extract_information
from gap_analysis.analyzer import find_research_gaps





from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="Research Gap Finder",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Gap Finder")
st.write(
    "Upload multiple research papers, ask questions "
    "and discover potential research gaps."
)


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "paper_analyses" not in st.session_state:
    st.session_state.paper_analyses = None


uploaded_files = st.file_uploader(
    "Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button("Process Research Papers"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:

        os.makedirs("uploads", exist_ok=True)

        file_paths = []

        for file in uploaded_files:

            unique_name = str(uuid.uuid4()) + ".pdf"

            path = os.path.join(
                "uploads",
                unique_name
            )

            with open(path, "wb") as f:
                f.write(file.getbuffer())

            file_paths.append(path)

        with st.spinner("Processing research papers..."):

            documents = load_papers(file_paths)

            chunks = split_documents(documents)

            # Create a unique collection for this session's
            # uploaded paper collection.
            st.session_state.vector_store = create_vector_store(
                chunks,
                collection_name="papers_" + uuid.uuid4().hex
            )

            analyses = []

            for path in file_paths:

                paper_name = os.path.basename(path)

                paper_docs = [
                    doc for doc in documents
                    if doc.metadata.get("paper_name") == paper_name
                ]

                paper_text = "\n".join(
                    doc.page_content for doc in paper_docs
                )

                # MVP limit; later replace with section-wise
                # map-reduce extraction for long papers.
                analysis = extract_information(
                    paper_text[:25000]
                )

                analyses.append({
                    "paper_name": paper_name,
                    "analysis": analysis.model_dump()
                })

            st.session_state.paper_analyses = analyses

        st.success("Research papers processed successfully!")


if st.session_state.vector_store:

    tab1, tab2 = st.tabs([
        "💬 Research Paper Q&A",
        "🔍 Research Gap Finder"
    ])

    with tab1:

        st.subheader("Ask questions across your papers")

        question = st.text_input(
            "Enter your question"
        )

        if st.button("Get Answer"):

            if question.strip():

                with st.spinner("Finding relevant information..."):

                    answer, sources = ask_question(
                        st.session_state.vector_store,
                        question
                    )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Supporting Sources")

                for source in sources:

                    with st.expander(
                        f"{source['paper']} | Page {source['page']}"
                    ):

                        st.write(source["content"])

    with tab2:

        st.subheader("Discover potential research gaps")

        if st.button("Analyze Research Gaps"):

            with st.spinner("Comparing research papers..."):

                gap_results = find_research_gaps(
                    json.dumps(
                        st.session_state.paper_analyses,
                        indent=2
                    )
                )

            st.markdown("### Potential Research Gaps")

            st.write(gap_results)

else:

    st.info("Upload and process papers to get started.")