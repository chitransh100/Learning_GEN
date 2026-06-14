from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import  RunnableLambda, RunnableBranch, RunnablePassthrough

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

prompt = PromptTemplate(
    template = "write me a detailed report on the topic {topic}",
    input_variables= ['topic']
)

prompt_2 = PromptTemplate(
    template = "write me the summary of the text {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain_1 = prompt | model | parser

chain_2 = RunnableBranch(
    (lambda x: count_words(x) > 500, prompt_2 | model | parser),
    RunnablePassthrough()
)

final_chain = chain_1 | chain_2

result = final_chain.invoke({'topic' : "black_hole"})

print(result);