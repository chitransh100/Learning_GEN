from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from dotenv import load_dotenv

load_dotenv();

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=1000,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

message = [
    SystemMessage("you are a helpfull assistent"),
    HumanMessage("Is it possible that i can be the worlds best coder")
]

result = model.invoke(message);
message.append(result);
print(result.content);