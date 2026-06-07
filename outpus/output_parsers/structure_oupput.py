from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="First fact about the topic"),
    ResponseSchema(name="fact_2", description="Second fact about the topic"),
    ResponseSchema(name="fact_3", description="Third fact about the topic"),
    ResponseSchema(name="fact_4", description="Fourth fact about the topic"),
    ResponseSchema(name="fact_5", description="Fifth fact about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template="""
Give me 5 facts about {topic}.

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

formatted_prompt = prompt.invoke({"topic": "black hole"})

result = model.invoke(formatted_prompt)

print("Raw Output:")
print(result.content)

final_result = parser.parse(result.content)

print("\nParsed Output:")
print(final_result)

chain = prompt | model | parser
result = chain.invoke({'topic':'black hole'});
print(result);