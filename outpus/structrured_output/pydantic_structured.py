from typing import Optional, Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

#model dont support pydantic 
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.5,
)

model = ChatHuggingFace(llm = llm)

class Review(BaseModel):
    keythemes: list[str] = Field(description="generate the key themes or the points discussed the in the review")
    summary: str = Field(description="generate a breif summary of the complete review")
    sentiment: Literal["positive", "neg", "neutral"] = Field(description="tell me about the sentimento of teh review either good, bad or neutral")
    pros: Optional[str] = Field(default=None, description="mention all teh pros if there")
    cons: Optional[str] = Field(default=None, description="mention all teh cons if there")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """
    I have been using the NovaNote AI workspace for roughly four months as part of my daily workflow, primarily for research, technical note-taking, project planning, and meeting documentation. My overall experience has been positive, but it has not been without frustrations, and I think the product still has several areas where it needs to mature before I could confidently recommend it to every type of user.

The biggest strength of the platform is its ability to organize large volumes of information. I often work with dozens of research papers, meeting transcripts, and design documents simultaneously, and NovaNote does an impressive job of connecting related concepts across different notes. The search functionality is particularly strong. In several instances, I was able to find references buried deep inside old notes that I had completely forgotten existed. This alone has saved me a significant amount of time.

Another aspect that deserves praise is the user interface. The layout is clean, modern, and generally intuitive. New users can become productive fairly quickly without spending hours learning the system. The onboarding process is one of the better ones I have seen in productivity software. Features are introduced gradually rather than overwhelming the user with dozens of options on day one.

That said, performance has been inconsistent. While simple note-taking operations are usually fast, the application occasionally becomes sluggish when working with larger knowledge bases containing thousands of documents. Search indexing can sometimes take longer than expected, and I have experienced a few situations where recently added notes did not immediately appear in search results. These issues are not catastrophic, but they do interrupt the workflow.

The AI-assisted summarization feature is another mixed area. On one hand, it produces surprisingly accurate summaries for technical documents and meeting notes. On the other hand, it occasionally omits important contextual details, leading to summaries that are technically correct but practically incomplete. This means I still need to manually verify critical information rather than trusting the generated output entirely.

Customer support has also been somewhat inconsistent. During one incident involving synchronization problems across devices, the support team responded within a few hours and resolved the issue efficiently. However, another ticket regarding export functionality remained unanswered for nearly a week. The quality of support therefore seems highly dependent on the nature of the problem and perhaps the support representative assigned to the case.

Pricing is another consideration. For individual professionals, the subscription cost is reasonable given the feature set. However, for students, hobbyists, or small teams operating on a limited budget, the pricing may feel difficult to justify, especially when several lower-cost competitors provide a subset of similar capabilities.

Security and privacy appear to be taken seriously. The company provides detailed documentation about data handling practices and offers granular permission controls. Although I appreciate this transparency, I would still like to see independent third-party security audits published more regularly to strengthen confidence in the platform.

Overall, NovaNote is a capable and thoughtfully designed productivity tool that excels at information organization, knowledge management, and AI-assisted workflows. Nevertheless, performance inconsistencies, occasional AI inaccuracies, and uneven customer support prevent it from being an exceptional product. I would recommend it to researchers, engineers, consultants, and knowledge workers who deal with large amounts of information,
    """);

print(result.sentiment);