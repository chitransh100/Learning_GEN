from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ------------------------------------------------------------------
# Load environment variables (optional)
# ------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------
# Create embedding model
# ------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------------
# Sample documents
# ------------------------------------------------------------------
documents = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
        metadata={"team": "Royal Challengers Bangalore"},
    ),
    Document(
        page_content="Rohit Sharma is a destructive opening batsman known for his elegant stroke play.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="MS Dhoni is a legendary captain known for his calm demeanor and finishing skills.",
        metadata={"team": "Chennai Super Kings"},
    ),
    Document(
        page_content="Jasprit Bumrah is one of the most lethal fast bowlers in IPL with an unorthodox action.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="Ravindra Jadeja is a versatile all-rounder excelling in both batting and bowling.",
        metadata={"team": "Chennai Super Kings"},
    ),
]

# ------------------------------------------------------------------
# Create / Load Chroma Vector Store
# ------------------------------------------------------------------
vector_store = Chroma(
    collection_name="ipl_players",
    embedding_function=embedding_model,
    persist_directory="./chroma_db",
)

# ------------------------------------------------------------------
# Add documents
# ------------------------------------------------------------------
document_ids = vector_store.add_documents(documents)

print("Document IDs:")
print(document_ids)

# ------------------------------------------------------------------
# View stored documents
# ------------------------------------------------------------------
all_docs = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nStored Documents:")
print(all_docs)

# ------------------------------------------------------------------
# Similarity Search
# ------------------------------------------------------------------
query = "Who is a bowler?"

results = vector_store.similarity_search(
    query=query,
    k=2
)

print("\nTop Matching Documents:")
for doc in results:
    print("-", doc.page_content)

# ------------------------------------------------------------------
# Similarity Search with Scores
# ------------------------------------------------------------------
scored_results = vector_store.similarity_search_with_score(
    query=query,
    k=2
)

print("\nSimilarity Scores:")
for doc, score in scored_results:
    print(f"Score: {score:.4f}")
    print(doc.page_content)
    print()

# ------------------------------------------------------------------
# Metadata Filtering
# ------------------------------------------------------------------
csk_players = vector_store.similarity_search(
    query="",
    filter={"team": "Chennai Super Kings"},
    k=10,
)

print("\nCSK Players:")
for doc in csk_players:
    print("-", doc.page_content)

# ------------------------------------------------------------------
# Update Document
# ------------------------------------------------------------------
updated_virat_doc = Document(
    page_content=(
        "Virat Kohli, former captain of Royal Challengers Bangalore, "
        "is renowned for his aggressive leadership and consistency."
    ),
    metadata={"team": "Royal Challengers Bangalore"},
)

virat_id = document_ids[0]

vector_store.update_document(
    document_id=virat_id,
    document=updated_virat_doc,
)

print("\nVirat Kohli document updated.")

# ------------------------------------------------------------------
# Delete Document
# ------------------------------------------------------------------
vector_store.delete(ids=[virat_id])

print("Virat Kohli document deleted.")

# ------------------------------------------------------------------
# Verify Deletion
# ------------------------------------------------------------------
remaining_docs = vector_store.get(include=["documents"])

print("\nRemaining Documents:")
for doc in remaining_docs["documents"]:
    print("-", doc)