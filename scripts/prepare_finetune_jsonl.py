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
JSON_INPUT = Path("fixed_remaining.json")
PHASE1_JSONL = Path("scjm_ft_phase1.jsonl")
TEST_JSONL = Path("test_phase1.jsonl")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_db"
COLLECTION_NAME_MAIN = "phase1_finetune"
COLLECTION_NAME_TEST = "phase1_test"

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
SYSTEM_MESSAGE = {"role": "system", "content": "You are a legal assistant expert in Indian Supreme Court judgments."}

def generate_qa_from_summary(record):
    case_title = record["id"].replace("_", " ")
    system_msg = SystemMessage(
        content="You are a legal assistant expert in Indian Supreme Court judgments. Generate concise Q&A for factual fine-tuning."
    )

    prompt = (
        f"Given the following case summary, generate exactly 3 pairs of question-answer messages to teach a model about this case.\n"
        f"Each pair should follow the OpenAI chat format with 'user' and 'assistant' roles and be returned as a JSON list of message lists.\n"
        f"Example format:\n"
        f"[ [{{\"role\": \"user\", \"content\": \"Q1\"}}, {{\"role\": \"assistant\", \"content\": \"A1\"}}], ... ]\n\n"
        f"SUMMARY:\n{record['summary']}\n"
        f"\nCASE TYPE: {record.get('case_type', '')}\nLEGAL ISSUES: {', '.join(record.get('legal_issues', []))}\n"
        f"ACTS CITED: {', '.join(record.get('acts_cited', []))}\nFINAL OUTCOME: {record.get('final_outcome', '')}"
    )

    messages = [system_msg, HumanMessage(content=prompt)]

    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip()

        # Strip Markdown code block wrapper if present
        if raw_output.startswith("```json"):
            raw_output = raw_output.removeprefix("```json").removesuffix("```").strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output.removeprefix("```").removesuffix("```").strip()

        print(f"🔁 Raw LLM output for {record['id']}:\n{raw_output}\n")

        if not raw_output:
            print(f"⚠️ Empty response from LLM for {record['id']}")
            return []

        qa_pairs = json.loads(raw_output)

        if not isinstance(qa_pairs, list):
            print(f"⚠️ LLM response for {record['id']} is not a list.")
            return []

        return qa_pairs

    except Exception as e:
        print(f"⚠️ Error calling LLM or parsing response for {record['id']}: {e}")
        return []

# --- Runner ---
def main(test_mode=False):
    with open(JSON_INPUT, "r", encoding="utf-8") as f:
        content = f.read()
        if content.strip().startswith("["):  # Proper JSON array
            records = json.loads(content)
        else:  # Newline-delimited JSON fallback (NDJSON)
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

    output_jsonl = TEST_JSONL if test_mode else PHASE1_JSONL
    mongo_col = col_test if test_mode else col_main

    test_id = "Ramakant_Ambalal_Choksi_vs_Harish_Ambalal_Choksi_on_22_November_2024_1"
    records_to_process = [r for r in records if r["id"] == test_id] if test_mode else records

    with open(output_jsonl, "a", encoding="utf-8") as out:
        for record in tqdm(records_to_process, desc="Generating fine-tune QA pairs"):
            qa_blocks = generate_qa_from_summary(record)
            for block in qa_blocks:
                item = {"messages": [SYSTEM_MESSAGE] + block}
                json.dump(item, out)
                out.write("\n")
                mongo_col.insert_one(item)

    print(f"✅ Done. Saved to {output_jsonl} and inserted into MongoDB collection '{mongo_col.name}'")

if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
