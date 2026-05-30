from langchain_core.prompts import PromptTemplate

template = """
You are an expert code assistant.

Your task is to summarize the given code repo.

### Instructions:
- Summary Type: {style}
- Style Detail: {sub_style}
- Length: {length}
- Focus Area: {focus}

### Guidelines:
- Keep the summary accurate and concise
- Do not add information not present in the text
- Highlight key ideas, methodology, and results

### Paper:
{paper}

### Output Format:
- Title
- Key Idea
- Summary
- Conclusion
"""

prompt_template = PromptTemplate(
    input_variables=["style", "sub_style", "length", "focus", "paper"],
    template=template,
    validate_template=True
)

prompt_template.save("code_assistant.json"); 