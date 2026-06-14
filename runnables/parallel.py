from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

prompt_1 = PromptTemplate(
    template= "write me a post for my twitter account on the topic {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template= "write me a post for my linkdin account on the topic {topic}"
)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt_1, model, parser) ,
    'linkdin' : prompt_2 | model | parser
})

result = chain.invoke({'topic' : 'AI'})

print(result)