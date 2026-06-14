from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import  RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

def count_words(text):
    return len(text.split())

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm = llm)

loader = PyPDFLoader("GATE-score.pdf");

doc = loader.load()

# print(len(doc))
print(doc[0].page_content)
print(doc[1].metadata)

