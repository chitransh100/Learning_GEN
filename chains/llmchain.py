from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM (using the same model as in conditional.py)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)
model = ChatHuggingFace(llm=llm)

# Define a prompt template
prompt = PromptTemplate(
    template="What is a good name for a company that makes {product}? Just give me 3 name options.",
    input_variables=["product"],
)

# Create the LLMChain
# Note: LLMChain is the legacy way to run a model with a prompt. 
# The modern LCEL equivalent would just be `chain = prompt | model`
# This won't run because in the newermodels there is no LLMChain this was a mistake that no longer exists
chain = LLMChain(llm=model, prompt=prompt)

if __name__ == "__main__":
    # Run the chain using invoke (commented out as requested)
    # result = chain.invoke({"product": "eco-friendly water bottles"})
    # print(result['text'])

    # Run the chain using run
    # Note: run() directly returns the string output rather than a dictionary
    result = chain.run(product="eco-friendly water bottles")
    print(result)
