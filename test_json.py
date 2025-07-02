import json

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded JSON file with {len(data)} items.")
        return data
    except json.JSONDecodeError as e:
        print("❌ JSON Decode Error:")
        print(f"  {e.msg} at line {e.lineno}, column {e.colno}")
    except Exception as e:
        print("❌ Failed to load file:")
        print(f"  {type(e).__name__}: {e}")

if __name__ == "__main__":
    load_json("fixed.json")
