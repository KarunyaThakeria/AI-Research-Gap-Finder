from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import EnsembleRetriever


def create_vector_store(chunks):

    # 1. Dense embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 2. Store dense vectors in ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_db",
        collection_name="research_papers"
    )

    # 3. Dense retriever
    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    # 4. Sparse retriever using BM25
    sparse_retriever = BM25Retriever.from_documents(
        chunks
    )

    sparse_retriever.k = 5

    # 5. Hybrid retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            dense_retriever,
            sparse_retriever
        ],
        weights=[0.5, 0.5]
    )

    return vector_store, ensemble_retriever