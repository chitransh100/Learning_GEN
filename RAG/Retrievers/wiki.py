from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "the geopolitical history of india and pakistan from the perspective of china"

docs = retriever.invoke(query)

for i, doc in enumerate(docs, start=1):
    print(f"----- Result {i} -----\n")
    print(f"Content:\n{doc.page_content[:500]}...\n")