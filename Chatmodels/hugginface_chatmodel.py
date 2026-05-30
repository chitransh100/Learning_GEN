from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv;

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("what are the rising tensions in the US and iran ")

print(result)