import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

HISTORY_FILE = "Chatmodels/chat_history.txt"

def load_history():
    """Loads chat history from the text file and converts it to Langchain message objects."""
    messages = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Human: "):
                    messages.append(HumanMessage(content=line[7:]))
                elif line.startswith("AI: "):
                    messages.append(AIMessage(content=line[4:]))
    return messages

def save_history(messages):
    """Saves the current list of Langchain message objects to the text file."""
    with open(HISTORY_FILE, "w") as f:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                f.write(f"Human: {msg.content}\n")
            elif isinstance(msg, AIMessage):
                f.write(f"AI: {msg.content}\n")

# 1. Create a ChatPromptTemplate with a MessagesPlaceholder
# This allows us to inject an entire list of past messages dynamically!

chat_template = ChatPromptTemplate.from_messages([
    ("system", """You only respond in valid JSON format.
                Never add any extra text outside the JSON.
                Structure every response as:
                {{
                    "answer": "...",
                    "confidence": "high/medium/low",
                    "source": "..."
                }}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 2. Setup the model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature=0.5,
)
model = ChatHuggingFace(llm=llm)

print("--- Chatbot Started ---")
print("(Type 'exit' to quit)\n")

# Load past chats when starting up
chat_history = load_history()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
        
    # 3. Format the prompt with the loaded chat history and current input
    prompt_value = chat_template.format_messages(
        chat_history=chat_history,
        user_input=user_input
    )
    
    # 4. Get response from model
    response = model.invoke(prompt_value)
    print(f"AI: {response.content}\n")
    
    # 5. Append new messages to our running history list
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    
    # 6. Save updated history to the file
    save_history(chat_history)