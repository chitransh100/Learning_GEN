from langchain_core.documents import Document
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.multi_query import MultiQueryRetriever

from transformers import pipeline

# ==================================================
# Hugging Face Embeddings
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# Sample Documents
# ==================================================

docs = [
    Document(
        page_content="""
        Climate change is causing rising global temperatures,
        extreme weather events, and melting glaciers.
        """
    ),
    Document(
        page_content="""
        Renewable energy sources such as solar and wind power
        are helping reduce dependence on fossil fuels.
        """
    ),
    Document(
        page_content="""
        Artificial Intelligence is transforming healthcare,
        education, and financial services.
        """
    ),
    Document(
        page_content="""
        Large Language Models can generate human-like text,
        summarize documents, and answer questions.
        """
    ),
    Document(
        page_content="""
        Electric vehicles are becoming increasingly popular
        due to lower emissions and government incentives.
        """
    ),
    Document(
        page_content="""
        Quantum computing has the potential to solve
        complex optimization problems much faster than
        classical computers.
        """
    ),
]

vector_store = FAISS.from_documents(
    docs,
    embeddings
)

# ==================================================
# Hugging Face LLM
# ==================================================

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=100
)

llm = HuggingFacePipeline(
    pipeline=generator
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 2}
    ),
    llm=llm
)

# ==================================================
# Query
# ==================================================

query = "How is AI changing industries?"

results = retriever.invoke(query)

# ==================================================
# Display Results
# ==================================================

for i, doc in enumerate(results, start=1):
    print(f"\n{'='*60}")
    print(f"Result {i}")
    print(f"{'='*60}")
    print(doc.page_content.strip())