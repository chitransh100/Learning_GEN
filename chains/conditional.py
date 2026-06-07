from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)
model = ChatHuggingFace(llm=llm)

prompt_1 = PromptTemplate(
    template = "just output the sentiment of the feedback: {text}",
    input_variable = ['text'],
)

parser = StrOutputParser();

branch_chain = RunnableBranch(
    (lambda x:x.sentiment = "positive ")
)