from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

# Define structure using Pydantic
class TopicFacts(BaseModel):
    fact_1: str = Field(description="First fact about the topic")
    fact_2: str = Field(description="Second fact about the topic")
    fact_3: str = Field(description="Third fact about the topic")
    fact_4: str = Field(description="Fourth fact about the topic")
    fact_5: str = Field(description="Fifth fact about the topic")

# Bind structured output to model
structured_model = model.with_structured_output(TopicFacts)

prompt = ChatPromptTemplate.from_template("Give me 5 facts about {topic}.")

# Chain it
chain = prompt | structured_model

result = chain.invoke({"topic": "black hole"})

print(result)           # TopicFacts object
print(result.fact_1)    # Access individual fields
print(result.model_dump())  # Convert to dict