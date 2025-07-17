import uuid
import time
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pymongo import MongoClient

# --- Load Environment ---
load_dotenv()

# --- Azure OpenAI Setup ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    temperature=0.3
)

# --- MongoDB Setup ---
client = MongoClient(os.getenv("MONGO_URI"))
db = client["legal_db"]
chat_col = db["legal_chat_history"]

SYSTEM_MESSAGE = SystemMessage(
    content="You are a legal assistant expert in Indian Supreme Court judgments."
)

# --- Chat History in MongoDB ---
class MongoChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    def add_message(self, message: BaseMessage):
        chat_col.insert_one({
            "session_id": self.session_id,
            "type": message.type,
            "content": message.content,
            "timestamp": time.time()
        })

    def get_messages(self) -> list[BaseMessage]:
        docs = list(chat_col.find({"session_id": self.session_id}).sort("timestamp"))
        messages = []
        for doc in docs:
            if doc["type"] == "human":
                messages.append(HumanMessage(content=doc["content"]))
            elif doc["type"] == "ai":
                messages.append(AIMessage(content=doc["content"]))
        return messages

    def clear(self):
        chat_col.delete_many({"session_id": self.session_id})

    @property
    def messages(self) -> list[BaseMessage]:
        return self.get_messages()

# --- Get Chat History ---
def get_session_history(session_id: str):
    return MongoChatMessageHistory(session_id=session_id)

# --- Simple QA Flow ---
def generate_legal_response(input_dict, config):
    session_id = config["configurable"]["session_id"]
    history = input_dict["history"]
    user_input = input_dict["input"]

    messages = [SYSTEM_MESSAGE] + history + [HumanMessage(content=user_input)]
    response = llm.invoke(messages)
    return response

# --- Runnable Chain ---
conversation_runnable = RunnableLambda(generate_legal_response)
runnable_with_history = RunnableWithMessageHistory(
    conversation_runnable,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- CLI Loop ---
def main():
    session_id = str(uuid.uuid4())
    print(f"\n[🧾 Legal Chat Session ID: {session_id}]")
    print("Type 'exit' to quit.")
    print("---------------------------------------------")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chat.")
            break

        response = runnable_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"Bot: {response.content}\n")

if __name__ == "__main__":
    main()
