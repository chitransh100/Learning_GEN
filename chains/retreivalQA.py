import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# 1. Create a sample text document for demonstration purposes
sample_text_path = "sample_data.txt"
if not os.path.exists(sample_text_path):
    with open(sample_text_path, "w") as f:
        f.write("LangChain is a framework for developing applications powered by language models.\n"
                "It enables applications that are context-aware and reason.\n"
                "RetrievalQA is a chain that does question answering over a set of documents.\n"
                "A vector store is used to index and search embeddings.")

# 2. Load the document
loader = TextLoader(sample_text_path)
documents = loader.load()

# 3. Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 4. Initialize embeddings (using a lightweight HuggingFace model)
# Note: This might download the embedding model on first run
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Create the vector store
# Note: This requires the `faiss-cpu` package. If you don't have it, run: `pip install faiss-cpu`
try:
    vectorstore = FAISS.from_documents(texts, embeddings)
except ImportError:
    print("Error: Please install FAISS to run this example.")
    print("You can install it by running: pip install faiss-cpu")
    exit()

# 6. Create the retriever from the vector store
retriever = vectorstore.as_retriever()

# 7. Initialize the LLM (using the same Llama model)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.1, # Lower temperature for more factual answering
)

# 8. Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" means it simply stuffs all retrieved docs into the prompt
    retriever=retriever,
    return_source_documents=True # This returns the documents that were used to find the answer
)

if __name__ == "__main__":
    # 9. Run the chain with a query
    query = "What is RetrievalQA?"
    print(f"Question: {query}\n")
    
    # Run the chain using invoke
    result = qa_chain.invoke({"query": query})
    
    print("Answer:")
    print(result['result'])
    
    print("\n-------------------------")
    print("Source Documents Used:")
    for i, doc in enumerate(result['source_documents']):
        print(f"Document {i+1}: {doc.page_content.strip()}")
