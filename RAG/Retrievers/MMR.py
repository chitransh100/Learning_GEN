from langchain_core.tools import retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Sample Documents
# --------------------------------------------------

docs = [
    Document(
        page_content="Virat Kohli is one of the most successful batsmen in IPL history.",
        metadata={"team": "RCB"}
    ),
    Document(
        page_content="Rohit Sharma is a destructive opening batsman.",
        metadata={"team": "MI"}
    ),
    Document(
        page_content="MS Dhoni is a legendary wicketkeeper and captain.",
        metadata={"team": "CSK"}
    ),
    Document(
        page_content="Jasprit Bumrah is one of the best fast bowlers in world cricket.",
        metadata={"team": "MI"}
    ),
    Document(
        page_content="Mohammed Shami is known for his seam bowling and accuracy.",
        metadata={"team": "India"}
    ),
    Document(
        page_content="Ravindra Jadeja is an all-rounder who bowls left-arm spin.",
        metadata={"team": "CSK"}
    ),
]

vector_store = FAISS.from_document(
    docs,
    embeddings
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.5
    }
)

query = "Tell me about bowlers"

results = retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    print(f"\n{'=' * 50}")
    print(f"Result {i}")
    print(f"{'=' * 50}")
    print(doc.page_content)
    print("Metadata:", doc.metadata)