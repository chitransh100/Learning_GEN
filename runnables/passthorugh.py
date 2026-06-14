from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = "create me a joke for topic {topic}",
    input_variables=['topic']
)
prompt_1 = PromptTemplate(
    template = "create me the explaination of the joke for topic {text}",
    input_variables=['text']
)

parser = StrOutputParser();

chain_1 = prompt | model | parser
chain_2 = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explain': RunnableSequence(prompt_1, model, parser)
})

final_chain = chain_1 | chain_2

result = final_chain.invoke({'topic':'disabled_person'})

print(result);