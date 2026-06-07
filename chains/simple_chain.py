from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv;
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template= "give me 5 facts about the topic {topic}",
    input_variables= ['topic']
)

chain = prompt | model

result = chain.invoke({'topic': 'black hole'})

print(result.content);

chain.get_graph().print_ascii();