from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# ==================================================
# Embeddings
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# Long Documents
# ==================================================

docs = [
    Document(
        page_content="""
        Diabetes is a chronic disease that affects how the body
        processes blood sugar. Common symptoms include increased
        thirst, frequent urination, fatigue, and blurred vision.

        Treatment often involves lifestyle modifications,
        blood sugar monitoring, insulin therapy, and oral
        medications. Researchers are continuously studying
        new therapies for better diabetes management.
        """
    ),

    Document(
        page_content="""
        Hypertension, also known as high blood pressure,
        increases the risk of heart disease and stroke.

        Symptoms are often not noticeable, which is why
        it is called the silent killer.

        Treatment includes reducing salt intake,
        exercising regularly, maintaining a healthy weight,
        and taking prescribed medications.
        """
    ),

    Document(
        page_content="""
        Asthma is a respiratory condition causing airway
        inflammation and breathing difficulties.

        Common symptoms include wheezing, coughing,
        chest tightness, and shortness of breath.

        Treatment often includes inhalers and avoiding
        environmental triggers.
        """
    ),

    Document(
        page_content="""
        Artificial Intelligence is transforming healthcare by
        improving diagnostics, predicting diseases, and helping
        physicians make data-driven decisions.

        AI systems can analyze medical images and patient records
        with increasing accuracy.
        """
    ),
]

vector_store = FAISS.from_documents(
    docs,
    embeddings
)

base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# ==================================================
# Hugging Face LLM
# ==================================================

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=100
)

llm = HuggingFacePipeline(
    pipeline=generator
)

# ==================================================
# Compressor
# ==================================================

compressor = LLMChainExtractor.from_llm(llm)

# ==================================================
# Contextual Compression Retriever
# ==================================================

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# ==================================================
# Query
# ==================================================

query = "What are the symptoms of diabetes?"

results = compression_retriever.invoke(query)

# ==================================================
# Display Results
# ==================================================

for i, doc in enumerate(results, start=1):
    print(f"\n{'='*60}")
    print(f"Result {i}")
    print(f"{'='*60}")
    print(doc.page_content)