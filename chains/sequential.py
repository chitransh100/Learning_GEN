from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv;
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

prompt_1 = PromptTemplate(
    template = "give me a breif summary of the topic {topic}",
    input_variables = ['topic']
)

prompt_2 = PromptTemplate(
    template = 'give me a 5 point summary of the content {content}',
    input_variable = ['content']
)

parser = StrOutputParser();

chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({'topic' : 'black hole'})

print(result)

# chain.get_graph().print_ascii();
