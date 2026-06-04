from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

# Define the desired JSON data structure using Pydantic
class TopicInfo(BaseModel):
    summary: str = Field(description="A brief summary of the topic")
    interesting_facts: list[str] = Field(description="A list of interesting facts about the topic")

# Set up the JSON output parser
json_parser = JsonOutputParser(pydantic_object=TopicInfo)

# Create a prompt that includes the JSON format instructions
template = PromptTemplate(
    template="Tell me about {topic}.\n\n{format_instructions}\n",
    input_variables=["topic"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

# Build the chain
chain = template | model | json_parser

print("Invoking chain...")
# Invoke the chain, the result will be parsed into a Python dictionary
result = chain.invoke({'topic': "black holes"})

print("\n--- Output ---")
print(result)
print(f"Type of result: {type(result)}")