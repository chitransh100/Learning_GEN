from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
#     task="text-generation",
#     max_new_tokens=500,
#     temperature=0.5,
# )

# model = ChatHuggingFace(llm=llm)

loader = PyPDFLoader('C:/Users/nites/Desktop/genai/Learning_GEN/RAG/document_loaders/materials/BTP_Presentation.pdf')
doc = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_documents(doc)

print(result)