from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model_1 = ChatHuggingFace(llm=llm1)
model_2 = ChatHuggingFace(llm=llm2)

prompt_1 = PromptTemplate(
    template = "give me a complete summary of the text {text}",
    input_variables = ['text']
)

prompt_2 = PromptTemplate(
    template = "create a 5 question quiz from this {text}",
    input_variables = ['text']
)

prompt_3 = PromptTemplate(
    template = "create a comibned result from these notes {notes} and quiz {quiz} where there should be 2 section one for quiz and other for the summary ",
    input_variables = ['notes','quiz']
)

parser = StrOutputParser();

parallel_chain = RunnableParallel(
    notes=prompt_1 | model_1 | parser,
    quiz=prompt_2 | model_2 | parser
)

last_chain = prompt_3 | model_1 | parser

chain = parallel_chain | last_chain | parser

text = """
Artificial Intelligence (AI) is a branch of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence. These tasks include learning, reasoning, problem-solving, understanding natural language, recognizing images, and making decisions.

Machine Learning (ML) is a subset of AI that enables computers to learn from data without being explicitly programmed. Instead of following fixed instructions, machine learning algorithms identify patterns in data and use those patterns to make predictions or decisions. Common applications include recommendation systems, spam detection, and fraud detection.

Deep Learning is a specialized area of machine learning that uses artificial neural networks with multiple layers. Deep learning has achieved remarkable success in fields such as computer vision, speech recognition, and natural language processing. Modern AI systems like ChatGPT are powered by large language models trained on massive datasets.

One of the most important developments in AI is Generative AI. Generative AI models can create new content, including text, images, audio, and video. These models learn patterns from existing data and generate outputs that resemble human-created content. Popular examples include ChatGPT, Gemini, Claude, and image-generation systems such as DALL-E.

Despite its benefits, AI also raises concerns regarding privacy, bias, job displacement, and ethical decision-making. Researchers and policymakers are actively working on frameworks and regulations to ensure AI systems are developed and used responsibly.

The future of AI is expected to bring significant advancements in healthcare, education, transportation, and scientific research. As AI technologies continue to evolve, understanding their capabilities and limitations becomes increasingly important for individuals and organizations alike.
"""
result = chain.invoke({'text' : text})

print(result);