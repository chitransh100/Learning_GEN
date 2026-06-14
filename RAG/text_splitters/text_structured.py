from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(
    "C:/Users/nites/Desktop/genai/Learning_GEN/RAG/document_loaders/materials/BTP_Presentation.pdf"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)