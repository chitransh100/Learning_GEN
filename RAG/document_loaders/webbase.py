from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

url = 'https://portfolio-chitransh-kumars-projects.vercel.app/'
loader = WebBaseLoader(url)
doc = loader.load()

# Combine page content into a single string (in case of multiple docs)
page_text = "\n".join(d.page_content for d in doc)

prompt = PromptTemplate(
    template="answer the question {question} for the text: {text}",
    input_variables=['question', 'text']
)

parser = StrOutputParser()
chain = prompt | model | parser

while True:
    question = input("question: ")
    if question == "exit":
        print("good to speak to you")
        break
    result = chain.invoke({'question': question, 'text': page_text})
    print(result)