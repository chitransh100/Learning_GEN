from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv;

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    # max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage("you are a great assistent")
]

#simple chatbot
while True:
    prompt = input("you: ");
    chat_history.append(HumanMessage(prompt));
    if prompt == "exit":
        break;
    result = model.invoke(chat_history);
    chat_history.append(result)
    print("AI: ",result.content)

print(chat_history);

