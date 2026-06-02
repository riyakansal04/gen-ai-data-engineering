import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(key):
    value = os.getenv(key)
    if not value:
        raise ValueError(f" Missing required environment variable: {key}")
    return value

# Gemini config
GEMINI_API_KEY  = get_env_var("GEMINI_API_KEY")
MODEL           = get_env_var("GEMINI_MODEL")
MAX_RETRIES     = int(get_env_var("GEMINI_MAX_RETRIES"))
WAIT_SECONDS    = int(get_env_var("GEMINI_WAIT_SECONDS"))
DB_PATH = get_env_var("DB_PATH")
RAG_CHUNK_SIZE = int(get_env_var("RAG_CHUNK_SIZE"))
RAG_TOP_K      = int(get_env_var("RAG_TOP_K"))

# Data Generation config
DATAGEN_NUM_ROWS = int(get_env_var("DATAGEN_NUM_ROWS"))

# File paths
PATHS = {
    "user_activity"     : "data/user_activity.json",
    "sample_csv"        : "data/sample.csv",
    "augmented_csv"     : "data/augmented_data.csv",
    "sample_pdf"        : "data/sample.pdf",
    "prompt_task1_1"    : "prompts/prompt1.txt",
    "prompt_task1_2"    : "prompts/prompt2.txt",
    "prompt_task1_3"    : "prompts/prompt3.txt",
    "prompt_task2"      : "prompts/prompt_task2.txt",
    "prompt_task3"      : "prompts/prompt_task3.txt",
    "prompt_task4"  : "prompts/prompt_task4.txt",
    "prompt_task5" : "prompts/prompt_task5.txt",
    "customers_json" : "data/customers.json",
    "sales_json"     : "data/sales.json",
    "prompt_task8" : "prompts/prompt_task8.txt",
    "eval_questions" : "data/eval_questions.json",
    "prompt_task9"   : "prompts/prompt_task9.txt",
}