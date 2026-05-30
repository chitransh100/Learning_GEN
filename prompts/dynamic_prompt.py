from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

st.header("📄 Code Summarizer")

paper_text = st.text_area("Paste your research paper content")

style = st.selectbox(
    "Select Summary Type",
    ["Technical", "Non-Technical", "Structured"]
)
if style == "Technical":
    sub_style = st.selectbox(
        "Select Technical Style",
        ["Detailed Analysis", "Methodology Focus", "Results Focus"]
    )

elif style == "Non-Technical":
    sub_style = st.selectbox(
        "Select Non-Technical Style",
        ["Simple Explanation", "ELI5", "Story Format"]
    )

else:
    sub_style = st.selectbox(
        "Select Structured Format",
        ["Bullet Points", "Executive Summary", "Key Takeaways"]
    )

length = st.selectbox(
    "Select Length",
    ["Short (100 words)", "Medium (250 words)", "Long (500+ words)"]
)

focus = st.text_input("Focus Area (optional)")

# template = """
# You are an expert code assistant.

# Your task is to summarize the given code repo.

# ### Instructions:
# - Summary Type: {style}
# - Style Detail: {sub_style}
# - Length: {length}
# - Focus Area: {focus}

# ### Guidelines:
# - Keep the summary accurate and concise
# - Do not add information not present in the text
# - Highlight key ideas, methodology, and results

# ### Paper:
# {paper}

# ### Output Format:
# - Title
# - Key Idea
# - Summary
# - Conclusion
# """

# prompt_template = PromptTemplate(
#     input_variables=["style", "sub_style", "length", "focus", "paper"],
#     template=template
# )

prompt_template = load_prompt('code_assistant.json')

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)


if st.button("Summarise") and paper_text:
    
    chain = prompt_template | model
    
    result = chain.invoke({
        "style": style,
        "sub_style": sub_style,
        "length": length,
        "focus": focus if focus else "General",
        "paper": paper_text
    })

    st.subheader("📌 Summary")
    st.write(result.content)