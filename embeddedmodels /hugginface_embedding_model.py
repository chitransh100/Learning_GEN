from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2");

doc = [
    "this is chitransh",
    "trying some embedding models"
]

text = "i am the king"


result = embedding.embed_query(text);

print(str(result))