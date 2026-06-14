from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Sample text containing multiple topics
text = """
Black holes are regions of spacetime where gravity is so strong that nothing,
not even light, can escape. Scientists believe black holes form when massive
stars collapse at the end of their life cycle.

There are different types of black holes, including stellar black holes,
supermassive black holes, and intermediate black holes. The supermassive
black hole at the center of our galaxy is called Sagittarius A*.

Machine learning is a branch of artificial intelligence that enables computers
to learn patterns from data. It is widely used in recommendation systems,
fraud detection, and image recognition.

Deep learning is a subset of machine learning that uses neural networks with
many layers. These models have achieved state-of-the-art results in computer
vision and natural language processing.

Hard work is often considered one of the most important factors in achieving
success. Consistent effort over time can lead to significant improvements in
skills and knowledge.
"""

# Convert text into a Document
docs = [Document(page_content=text)]

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Semantic chunker
splitter = SemanticChunker(
    embeddings=embeddings
)

# Create semantic chunks
chunks = splitter.split_documents(docs)

print(f"\nTotal Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print("\n" + "=" * 60)
    print(f"CHUNK {i}")
    print("=" * 60)
    print(chunk.page_content)