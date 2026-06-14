from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import  RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_community.document_loaders import TextLoader

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

loader = TextLoader('text.txt');

docs = loader.load()

# print(type(docs))
# print(len(docs))
# print(type(docs[0]))
# print(docs[0].metadata)
# print(docs[0].page_content)

prompt = PromptTemplate(
    template= "write me a summary of the doc {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'text' : docs[0].page_content})

print(result);