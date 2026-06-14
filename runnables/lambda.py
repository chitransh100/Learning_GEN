from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import  RunnableParallel, RunnablePassthrough, RunnableLambda

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
    template = "give me a joke about the topic {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

chain_1 = prompt | model | parser

chain_2 = RunnableParallel({
    "joke" : RunnablePassthrough(),
    'words' : RunnableLambda(count_words)
})

final_chain = chain_1 | chain_2

result = final_chain.invoke({'topic' : 'race'})

fresult = """ {} \n word count - {} """.format(result['joke'], result['words'])
print(fresult)

