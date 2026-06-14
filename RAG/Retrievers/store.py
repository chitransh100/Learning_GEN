from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is a destructive opening batsman and captain of Mumbai Indians.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni is a legendary wicketkeeper batsman and captain.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is one of the best fast bowlers in world cricket.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is an excellent all-rounder who contributes with both bat and ball.",
        metadata={"team": "Chennai Super Kings"}
    )
]

# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------
vector_store = Chroma(
    collection_name="ipl_players",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# --------------------------------------------------
# Add Documents
# --------------------------------------------------
ids = vector_store.add_documents(docs)

print("Documents Added:")
print(ids)

# --------------------------------------------------
# Create Retriever
# --------------------------------------------------
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# Query
# --------------------------------------------------
query = "Who is a fast bowler?"

results = retriever.invoke(query)

# --------------------------------------------------
# Display Results
# --------------------------------------------------
for i, doc in enumerate(results, start=1):
    print(f"\n{'='*50}")
    print(f"Result {i}")
    print(f"{'='*50}")
    print(doc.page_content)
    print("Metadata:", doc.metadata)