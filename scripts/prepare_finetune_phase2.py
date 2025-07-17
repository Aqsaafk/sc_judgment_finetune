import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# --- Config ---
JSON_INPUT = Path("fixed.json")
PHASE2_JSONL = Path("scjm_ft_phase2.jsonl")
TEST_JSONL = Path("test_phase2.jsonl")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_db"
COLLECTION_NAME_MAIN = "phase2_finetune"
COLLECTION_NAME_TEST = "phase2_test"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Mongo Setup ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col_main = db[COLLECTION_NAME_MAIN]
col_test = db[COLLECTION_NAME_TEST]

# --- LLM Setup ---
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version="2024-02-15-preview",
    temperature=0.3
)

# --- Prompting ---
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a helpful and reliable legal assistant trained on Indian Supreme Court judgments. Your task is to answer legal questions factually and clearly based on the provided case details."
}


def generate_reasoning_convo(record):
    system_msg = SystemMessage(content=SYSTEM_MESSAGE["content"])
    prompt = (
        f"Given the following Supreme Court judgment summary, simulate a multi-turn chat between a user and an assistant.\n"
        f"The user first asks a factual question. The assistant answers. Then the user asks a follow-up reasoning question.\n"
        f"Return the full conversation as a single JSON object under 'messages', including 'weight': 0 for factual and 1 for reasoning replies.\n"
        f"SUMMARY:\n{record['summary']}\n\nLEGAL ISSUES: {', '.join(record.get('legal_issues', []))}\nFINAL OUTCOME: {record.get('final_outcome', '')}"
    )

    messages = [system_msg, HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip()

        if raw_output.startswith("```json"):
            raw_output = raw_output.removeprefix("```json").removesuffix("```").strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output.removeprefix("```").removesuffix("```").strip()

        print(f"\n🔁 Raw LLM output for {record['id']}:\n{raw_output}\n")

        convo = json.loads(raw_output)
        if not isinstance(convo, dict) or "messages" not in convo:
            print("⚠️ Invalid response format.")
            return None
        convo["messages"].insert(0, SYSTEM_MESSAGE)
        return convo

    except Exception as e:
        print(f"⚠️ Error calling LLM or parsing response for {record['id']}: {e}")
        return None

# --- Runner ---
def main(test_mode=False):
    with open(JSON_INPUT, "r", encoding="utf-8") as f:
        content = f.read()
        if content.strip().startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

    output_jsonl = TEST_JSONL if test_mode else PHASE2_JSONL
    mongo_col = col_test if test_mode else col_main

    test_id = "Ramakant_Ambalal_Choksi_vs_Harish_Ambalal_Choksi_on_22_November_2024_1"
    records_to_process = [r for r in records if r["id"] == test_id] if test_mode else records

    with open(output_jsonl, "a", encoding="utf-8") as out:
        for record in tqdm(records_to_process, desc="Generating reasoning conversations"):
            convo = generate_reasoning_convo(record)
            if convo:
                json.dump(convo, out)
                out.write("\n")
                mongo_col.insert_one(convo)

    print(f"✅ Done. Saved to {output_jsonl} and inserted into MongoDB collection '{mongo_col.name}'")

if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
