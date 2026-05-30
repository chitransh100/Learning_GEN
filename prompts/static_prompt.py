from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv;

import streamlit as st

load_dotenv()

st.header("Research Tool")

user_input = st.text_input("Enter your prompt here")

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=1000,
    temperature=0.5,
)
model = ChatHuggingFace(llm=llm)

if st.button("summarise") and user_input:
    result = model.invoke(user_input)
    st.write(result.content);

# result = model.invoke("what are the rising tensions in the US and iran ")

# print(result)