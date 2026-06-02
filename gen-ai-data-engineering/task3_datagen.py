from google import genai
import os
import time
import logging
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from config import (
    GEMINI_API_KEY, MODEL, MAX_RETRIES,
    WAIT_SECONDS, DATAGEN_NUM_ROWS, PATHS
)

load_dotenv()

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)

# STEP 1 — Read CSV file
def load_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"CSV file is empty: {filepath}")
    logger.info(f"Loaded CSV from: {filepath}")
    logger.info(f"Original data: {len(df)} rows x {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df

# STEP 2 — Load prompt template and fill placeholders
def load_prompt_template(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt template not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Prompt template is empty: {filepath}")
    logger.info(f"Loaded prompt template from: {filepath}")
    return content

def build_prompt(df):
    csv_content = df.to_csv(index=False)
    start_id = df["employee_id"].max() + 1

    template = load_prompt_template(PATHS["prompt_task3"])
    prompt = template.replace("{csv_content}", csv_content)
    prompt = prompt.replace("{num_rows}", str(DATAGEN_NUM_ROWS))
    prompt = prompt.replace("{start_id}", str(start_id))
    return prompt, start_id

# STEP 3 — Call Gemini with retry logic
def call_gemini(prompt):
    logger.info(f"Sending prompt to Gemini ({MODEL})...")
    logger.info(f"Requesting {DATAGEN_NUM_ROWS} new synthetic rows...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            logger.info(f"Gemini responded on attempt {attempt}")
            return response.text

        except Exception as e:
            error_msg = str(e)

            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                if attempt < MAX_RETRIES:
                    wait = WAIT_SECONDS * attempt
                    logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed — server busy")
                    logger.warning(f"Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Gemini unavailable after {MAX_RETRIES} retries")

            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise RuntimeError("API quota exhausted — check your API key")

            else:
                raise RuntimeError(f"Unexpected error: {error_msg}")

# STEP 4 — Parse generated CSV rows
def parse_generated_rows(response_text, original_df):
    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("csv"):
            cleaned = cleaned[3:]
    cleaned = cleaned.strip()

    try:
        new_df = pd.read_csv(
            StringIO(cleaned),
            header=None,
            names=original_df.columns
        )
    except Exception as e:
        raise RuntimeError(f"Failed to parse generated rows: {e}\nRaw: {cleaned}")

    if new_df.empty:
        raise ValueError("Gemini returned no data rows")

    logger.info(f"Parsed {len(new_df)} generated rows successfully")
    return new_df

# STEP 5 — Validate generated data
def validate_generated_data(new_df):
    logger.info("Validating generated data...")
    issues = []

    if len(new_df) != DATAGEN_NUM_ROWS:
        issues.append(f"Expected {DATAGEN_NUM_ROWS} rows, got {len(new_df)}")

    null_counts = new_df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            issues.append(f"Column '{col}' has {count} null values")

    if not new_df["age"].between(18, 65).all():
        issues.append("Some ages are outside realistic range 18-65")

    if not (new_df["salary"] > 0).all():
        issues.append("Some salary values are not positive")

    if not new_df["years_experience"].between(0, 45).all():
        issues.append("Some experience values are outside realistic range")

    if issues:
        for issue in issues:
            logger.warning(f"Validation issue: {issue}")
    else:
        logger.info("All validation checks passed!")

    return issues

# MAIN
def main():
    logger.info("TASK 3 — DATA GENERATION & AUGMENTATION")

    original_df = load_csv(PATHS["sample_csv"])

    prompt, start_id = build_prompt(original_df)
    logger.info(f"Prompt built — new IDs will start from {start_id}")

    raw_response = call_gemini(prompt)

    new_df = parse_generated_rows(raw_response, original_df)

    validate_generated_data(new_df)

    combined_df = pd.concat([original_df, new_df], ignore_index=True)

    combined_df.to_csv(PATHS["augmented_csv"], index=False)
    logger.info(f"Saved augmented data to: {PATHS['augmented_csv']}")

    logger.info(f"Original rows  : {len(original_df)}")
    logger.info(f"Generated rows : {len(new_df)}")
    logger.info(f"Total rows     : {len(combined_df)}")

    print("\n--- ORIGINAL DATA ---")
    print(original_df.to_string(index=False))

    print("\n--- GENERATED DATA ---")
    print(new_df.to_string(index=False))

    print("\n--- COMBINED DATA ---")
    print(combined_df.to_string(index=False))

    logger.info("Task 3 completed successfully!")

if __name__ == "__main__":
    main()