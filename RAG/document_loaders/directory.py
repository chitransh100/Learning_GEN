from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import  RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

loader = DirectoryLoader(
    path = 'materials',
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

doc = loader.lazy_load()

# print(doc[14].page_content)
# print(len(doc))

for something in doc:
    print(something.metadata)
