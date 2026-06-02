from google import genai
import os
import time
from dotenv import load_dotenv

load_dotenv()

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

# LOAD PROMPTS FROM FILES — nothing hardcoded
def load_prompt(filename):
    filepath = os.path.join("prompts", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {filepath}")
    print(f" Loaded prompt from: {filepath}")
    return content

# CALL GEMINI — with retry + fail loudly
def call_gemini(prompt, prompt_name):
    print(f"\n Sending '{prompt_name}' to Gemini...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            print(f" '{prompt_name}' succeeded on attempt {attempt}")
            return response.text

        except Exception as e:
            error_msg = str(e)

            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                if attempt < MAX_RETRIES:
                    print(f"  Attempt {attempt}/{MAX_RETRIES} failed — server busy")
                    print(f" Waiting {WAIT_SECONDS}s before retry...")
                    time.sleep(WAIT_SECONDS)
                else:
                    raise RuntimeError(
                        f" '{prompt_name}' failed after {MAX_RETRIES} retries — server unavailable"
                    )

            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise RuntimeError(
                    f" '{prompt_name}' failed — API quota exhausted. Check your API key."
                )

            else:
                raise RuntimeError(
                    f" '{prompt_name}' failed — Unexpected error: {error_msg}"
                )

# MAIN
def main():
    print("   TASK 1 — PROMPT ENGINEERING FOR DATA ENGINEERING")

    # Load all prompts from files
    prompts = [
        ("PROMPT 1 - SQL Query Optimization",        "prompt1.txt"),
        ("PROMPT 2 - Fault Tolerant Pipeline Design", "prompt2.txt"),
        ("PROMPT 3 - Data Quality Rules Generation",  "prompt3.txt"),
    ]

    results = {}
    failed_prompts = []

    for title, filename in prompts:
        print(f"  {title}")

        try:
            # Step 1 — Load prompt from file
            prompt = load_prompt(filename)

            # Step 2 — Send to Gemini
            response_text = call_gemini(prompt, title)

            # Step 3 — Store and print response
            results[title] = response_text
            print("\n--- GEMINI RESPONSE ---")
            print(response_text)

        except (FileNotFoundError, ValueError) as e:
            # File missing or empty — fail loudly
            failed_prompts.append(title)
            print(f"\n FILE ERROR: {e}")

        except RuntimeError as e:
            # Gemini API error — fail loudly
            failed_prompts.append(title)
            print(f"\n API ERROR: {e}")

        # Wait between prompts
        if title != prompts[-1][0]:
            print(f"\n⏳ Waiting {WAIT_SECONDS}s before next prompt...")
            time.sleep(WAIT_SECONDS)

    # FINAL SUMMARY
    print("  FINAL SUMMARY")
    print(f" Successful : {len(results)}/{len(prompts)}")
    print(f" Failed     : {len(failed_prompts)}/{len(prompts)}")

    if failed_prompts:
        print("\n FAILED PROMPTS — must be resolved:")
        for fp in failed_prompts:
            print(f" {fp}")
        raise SystemExit("Task 1 incomplete — resolve errors above!")
    else:
        print("\n Task 1 completed successfully!")

if __name__ == "__main__":
    main()