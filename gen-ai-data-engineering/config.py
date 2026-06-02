
import os
from dotenv import load_dotenv
 
load_dotenv()
 
def get_env_var(key, default=None):
    """Get environment variable with optional default — doesn't crash on startup."""
    value = os.getenv(key)
    if not value:
        if default is None:
            print(f"⚠️  Warning: {key} not set, using default or will fail later if needed")
        return default
    return value
 
# ============================================================
# GEMINI CONFIG — Safe defaults that don't crash on startup
# ============================================================
GEMINI_API_KEY  = get_env_var("GEMINI_API_KEY", "")
MODEL           = get_env_var("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_RETRIES     = int(get_env_var("GEMINI_MAX_RETRIES", "3"))
WAIT_SECONDS    = int(get_env_var("GEMINI_WAIT_SECONDS", "30"))
 
# ============================================================
# DATABASE CONFIG
# ============================================================
DB_PATH = get_env_var("DB_PATH", "data/sales.db")
 
# ============================================================
# RAG CONFIG
# ============================================================
RAG_CHUNK_SIZE = int(get_env_var("RAG_CHUNK_SIZE", "200"))
RAG_TOP_K      = int(get_env_var("RAG_TOP_K", "3"))
 
# ============================================================
# DATA GENERATION CONFIG
# ============================================================
DATAGEN_NUM_ROWS = int(get_env_var("DATAGEN_NUM_ROWS", "10"))
 
# ============================================================
# FILE PATHS — These should exist in your project
# ============================================================
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
    "prompt_task4"      : "prompts/prompt_task4.txt",
    "prompt_task5"      : "prompts/prompt_task5.txt",
    "customers_json"    : "data/customers.json",
    "sales_json"        : "data/sales.json",
    "prompt_task8"      : "prompts/prompt_task8.txt",
    "eval_questions"    : "data/eval_questions.json",
    "prompt_task9"      : "prompts/prompt_task9.txt",
}
 
# ============================================================
# VALIDATION — Only check critical vars when actually needed
# ============================================================
def validate_api_key():
    """Call this only when making API calls, not on startup."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required to use Gemini.")
    return True