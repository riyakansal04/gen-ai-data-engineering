from google import genai
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG — all from .env, nothing hardcoded
# ============================================================
def get_env_var(key):
    value = os.getenv(key)
    if not value:
        raise ValueError(f" Missing required environment variable: {key}")
    return value

GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
MODEL          = get_env_var("GEMINI_MODEL")
MAX_RETRIES    = int(get_env_var("GEMINI_MAX_RETRIES"))
WAIT_SECONDS   = int(get_env_var("GEMINI_WAIT_SECONDS"))

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# STEP 1 — Load user activity from file
# ============================================================
def load_user_activity(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" Input file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "user_activity" not in data:
        raise ValueError(" Input file missing 'user_activity' key")
    print(f" Loaded user activity from: {filepath}")
    return data["user_activity"]

# ============================================================
# STEP 2 — Load prompt template and fill placeholders
# ============================================================
def load_prompt_template(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" Prompt template not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f" Prompt template is empty: {filepath}")
    print(f" Loaded prompt template from: {filepath}")
    return content

def build_prompt(activities):
    activity_text = "\n".join([
        f"- {a['user']} {a['action']}"
        + (f" worth ${a['amount']}" if a["amount"] > 0 else "")
        for a in activities
    ])

    template = load_prompt_template("prompts/prompt_task2.txt")
    prompt = template.replace("{activity_text}", activity_text)
    prompt = prompt.replace("{total_users}", str(len(activities)))
    return prompt

# ============================================================
# STEP 3 — Call Gemini with retry logic
# ============================================================
def call_gemini(prompt):
    print(f"\n Sending prompt to Gemini ({MODEL})...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            print(f"✅ Gemini responded on attempt {attempt}")
            return response.text

        except Exception as e:
            error_msg = str(e)

            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                if attempt < MAX_RETRIES:
                    wait = WAIT_SECONDS * attempt
                    print(f"  Attempt {attempt}/{MAX_RETRIES} failed — server busy")
                    print(f" Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f" Gemini unavailable after {MAX_RETRIES} retries")

            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise RuntimeError(" API quota exhausted — check your API key")

            else:
                raise RuntimeError(f" Unexpected error: {error_msg}")

# ============================================================
# STEP 4 — Parse and validate JSON response
# ============================================================
def parse_response(response_text):
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        required_keys = ["summary", "total_users", "purchasing_users", "total_revenue", "insights"]
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            raise ValueError(f" Missing keys in response: {missing}")

        print("JSON parsed and validated successfully")
        return parsed

    except json.JSONDecodeError as e:
        raise RuntimeError(f" Gemini returned invalid JSON: {e}\nRaw response: {response_text}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("   TASK 2 — PROGRAMMATIC LLM CHAT + JSON OUTPUT")

    # Step 1 — Load data
    activities = load_user_activity("data/user_activity.json")
    print(f"\n Loaded {len(activities)} user activity records")

    # Step 2 — Build prompt
    prompt = build_prompt(activities)
    print(" Prompt built successfully")

    # Step 3 — Call Gemini
    raw_response = call_gemini(prompt)

    # Step 4 — Parse JSON
    result = parse_response(raw_response)

    # Step 5 — Display structured JSON
    print("   STRUCTURED JSON OUTPUT")
    print(json.dumps(result, indent=2))

    # Step 6 — Display human-readable summary
    print("   INSIGHTS SUMMARY")
    print(f" Summary       : {result['summary']}")
    print(f" Total Users   : {result['total_users']}")
    print(f" Buyers        : {result['purchasing_users']}")
    print(f" Total Revenue : ${result['total_revenue']}")
    print(f"\n Insights:")
    for i, insight in enumerate(result["insights"], 1):
        print(f"   {i}. {insight}")

    print("\n Task 2 completed successfully!")

if __name__ == "__main__":
    main()