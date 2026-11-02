
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def create_vector_store(chunks, collection_name="research_papers"):

    # 1. Generate dense embeddings
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    # 2. Store embeddings in ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_db",
        collection_name=collection_name
    )

    # 3. Dense semantic retriever
    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    # 4. Sparse keyword retriever using BM25
    sparse_retriever = BM25Retriever.from_documents(
        chunks
    )

    sparse_retriever.k = 5

    # 5. Combine both retrievers using Ensemble Retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            dense_retriever,
            sparse_retriever
        ],
        weights=[0.5, 0.5]
    )

    # Return both objects
    return vector_store, ensemble_retriever
