from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnablePassthrough
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)
model = ChatHuggingFace(llm=llm)

prompt_1 = PromptTemplate(
    template = "just output the sentiment of the feedback: {text}",
    input_variable = ['text'],
)

parser = StrOutputParser()

positive_feedback_template = PromptTemplate(
    template="Write a thank you note for this positive feedback: {text}",
    input_variables=["text"],
)

negative_feedback_template = PromptTemplate(
    template="Write an apology note for this negative feedback: {text}",
    input_variables=["text"],
)

neutral_feedback_template = PromptTemplate(
    template="Write a simple acknowledgement for this neutral feedback: {text}",
    input_variables=["text"],
)

positive_chain = positive_feedback_template | model | parser
negative_chain = negative_feedback_template | model | parser
neutral_chain = neutral_feedback_template | model | parser

branch_chain = RunnableBranch(
    (lambda x: "positive" in x["sentiment"].lower(), positive_chain),
    (lambda x: "negative" in x["sentiment"].lower(), negative_chain),
    neutral_chain
)

full_chain = (
    RunnablePassthrough.assign(sentiment=prompt_1 | model | parser)
    | branch_chain
)

if __name__ == "__main__":
    result = full_chain.invoke({"text": "I really love your product, it's amazing!"})
    print(result)