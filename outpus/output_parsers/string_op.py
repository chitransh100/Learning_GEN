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

#prompt 1 
template_1 = PromptTemplate(
    template = 'write a detailed report on {topic}',
    input_variables = ['topic']
)
#prompt 2
template_2 = PromptTemplate(
    template = 'write a 5 line summary for the following text \n {text}',
    input_variables = ['text']
)

parser = StrOutputParser();

# prompt1 = template_1.invoke({'topic':'black hole'})
# result1 = model.invoke(prompt1);
# print(result1.content);
# print("\n now 2nd prompt");
# prompt2 = template_2.invoke({'text': result1.content})
# result2 = model.invoke(prompt2);

chain = template_1 | model | parser | template_2 | model | parser

chain.invoke({'topic': "black_hole"});