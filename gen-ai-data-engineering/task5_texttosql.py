from google import genai
import os
import json
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from tabulate import tabulate
from dotenv import load_dotenv
from config import (
    GEMINI_API_KEY, MODEL, MAX_RETRIES,
    WAIT_SECONDS, PATHS, DB_PATH
)

load_dotenv()

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# SCHEMA — used in prompt
# ============================================================
SCHEMA = """
customer (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT,
    email       TEXT,
    join_date   TEXT
)

sales (
    sale_id     INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product     TEXT,
    amount      REAL,
    sale_date   TEXT,
    FOREIGN KEY(customer_id) REFERENCES customer(customer_id)
)
"""

# ============================================================
# STEP 1 — Create and populate database
# ============================================================
def setup_database():
    os.makedirs("data", exist_ok=True)

    # Load data from files
    customers_path = PATHS["customers_json"]
    sales_path     = PATHS["sales_json"]

    if not os.path.exists(customers_path):
        raise FileNotFoundError(f"Customers file not found: {customers_path}")
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"Sales file not found: {sales_path}")

    with open(customers_path, "r", encoding="utf-8") as f:
        customers = json.load(f)["customers"]

    with open(sales_path, "r", encoding="utf-8") as f:
        sales_data = json.load(f)["sales"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS customer;

        CREATE TABLE customer (
            customer_id INTEGER PRIMARY KEY,
            name        TEXT,
            email       TEXT,
            join_date   TEXT
        );

        CREATE TABLE sales (
            sale_id     INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product     TEXT,
            amount      REAL,
            sale_date   TEXT,
            FOREIGN KEY(customer_id) REFERENCES customer(customer_id)
        );
    """)

    # Insert customers
    cursor.executemany(
        "INSERT INTO customer VALUES (:customer_id, :name, :email, :join_date)",
        customers
    )

    # Insert sales with dynamic dates
    today = datetime.now()
    for sale in sales_data:
        cursor.execute(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
            (
                sale["sale_id"],
                sale["customer_id"],
                sale["product"],
                sale["amount"],
                (today - timedelta(days=sale["days_ago"])).strftime("%Y-%m-%d")
            )
        )

    conn.commit()
    conn.close()
    logger.info(f"Database created: {DB_PATH}")
    logger.info(f"Inserted {len(customers)} customers and {len(sales_data)} sales")

# ============================================================
# STEP 2 — Load prompt template
# ============================================================
def load_prompt_template(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt template not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Prompt template is empty: {filepath}")
    logger.info(f"Loaded prompt template from: {filepath}")
    return content

def build_prompt(template, question):
    prompt = template.replace("{schema}", SCHEMA)
    prompt = prompt.replace("{question}", question)
    return prompt

# ============================================================
# STEP 3 — Call Gemini with retry logic
# ============================================================
def call_gemini(prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
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

# ============================================================
# STEP 4 — Clean SQL returned by Gemini
# ============================================================
def clean_sql(raw_sql):
    cleaned = raw_sql.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:]
    cleaned = cleaned.strip()
    logger.info(f"Generated SQL: {cleaned}")
    return cleaned

# ============================================================
# STEP 5 — Execute SQL on database
# ============================================================
def execute_sql(sql):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return rows, columns

    except sqlite3.Error as e:
        raise RuntimeError(f"SQL execution error: {e}\nSQL: {sql}")

    finally:
        conn.close()

# ============================================================
# STEP 6 — Format and display results
# ============================================================
def display_results(rows, columns, question, sql):
    print("\n" + "=" * 60)
    print(f"❓ Question : {question}")
    print(f"🔍 SQL      : {sql}")
    print("=" * 60)

    if not rows:
        print("📭 No results found for the query.")
    else:
        print(f"📊 Results  : {len(rows)} row(s) found\n")
        print(tabulate(rows, headers=columns, tablefmt="pretty"))

    print("-" * 60)

# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 50)
    logger.info("TASK 5 - TEXT TO SQL")
    logger.info("=" * 50)

    # Step 1 — Setup database
    setup_database()

    # Step 2 — Load prompt template
    template = load_prompt_template(PATHS["prompt_task5"])

    print("\n" + "=" * 60)
    print("   TEXT TO SQL - ASK QUESTIONS IN PLAIN ENGLISH")
    print("   Type 'exit' to quit")
    print("=" * 60)
    print("\nDatabase ready with customer and sales tables!")
    print("Example: 'highest sales amount done by a customer in last 3 days'")
    input("\nPress Enter to start...\n")

    # Step 3 — Interactive loop
    while True:
        question = input(" Your question: ").strip()

        if not question:
            print("  Please enter a question\n")
            continue

        if question.lower() == "exit":
            print("\n Exiting Text to SQL. Goodbye!")
            logger.info("User exited Text to SQL session")
            break

        try:
            # Generate SQL
            prompt = build_prompt(template, question)
            logger.info(f"Sending question to Gemini: {question}")
            raw_sql = call_gemini(prompt)

            # Clean SQL
            sql = clean_sql(raw_sql)

            # Execute SQL
            rows, columns = execute_sql(sql)

            # Display results
            display_results(rows, columns, question, sql)

        except RuntimeError as e:
            logger.error(f"Error: {e}")
            print(f" Error: {e}\n")

if __name__ == "__main__":
    main()