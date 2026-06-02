from google import genai
import os
import time
import logging
import PyPDF2
import docx
from dotenv import load_dotenv
from config import (
    GEMINI_API_KEY, MODEL, MAX_RETRIES,
    WAIT_SECONDS, PATHS
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
# STEP 1 — Extract text from documents
# ============================================================
def extract_pdf_text(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF file not found: {filepath}")
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        if len(reader.pages) == 0:
            raise ValueError(f"PDF has no pages: {filepath}")
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n{page_text}"
    if not text.strip():
        raise ValueError(f"Could not extract any text from PDF: {filepath}")
    logger.info(f"Extracted text from PDF: {filepath}")
    logger.info(f"Total characters extracted: {len(text)}")
    return text

def extract_word_text(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Word file not found: {filepath}")
    document = docx.Document(filepath)
    text = "\n".join([para.text for para in document.paragraphs if para.text.strip()])
    if not text.strip():
        raise ValueError(f"Could not extract text from Word doc: {filepath}")
    logger.info(f"Extracted text from Word doc: {filepath}")
    logger.info(f"Total characters extracted: {len(text)}")
    return text

def extract_document_text(filepath):
    """Auto-detect file type and extract text"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_pdf_text(filepath)
    elif ext in [".docx", ".doc"]:
        return extract_word_text(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or Word document.")

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

def build_prompt(template, document_content, question):
    prompt = template.replace("{document_content}", document_content)
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
# MAIN
# ============================================================
def main():
    logger.info("TASK 4 - DOCUMENT Q&A")


    print("   DOCUMENT Q&A - ASK ANYTHING ABOUT THE DOCUMENT")

    print(f"\nDefault document: {PATHS['sample_pdf']}")
    print("Supported formats: PDF (.pdf), Word (.docx, .doc)")
    custom_path = input("\nEnter document path (or press Enter to use default): ").strip()

    filepath = custom_path if custom_path else PATHS["sample_pdf"]

    # Extract text
    try:
        document_content = extract_document_text(filepath)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load document: {e}")
        print(f"\nError: {e}")
        return

    # Load prompt template
    template = load_prompt_template(PATHS["prompt_task4"])

    print(f"\n Document loaded: {filepath}")
    print(f"Characters extracted: {len(document_content):,}")
    print("\nType 'exit' to quit\n")
    input("Press Enter to start asking questions...\n")

    # Interactive Q&A loop
    while True:
        question = input(" Your question: ").strip()

        if not question:
            print("Please enter a question\n")
            continue

        if question.lower() == "exit":
            print("\n Exiting Document Q&A. Goodbye!")
            logger.info("User exited Q&A session")
            break

        prompt = build_prompt(template, document_content, question)

        try:
            logger.info(f"Sending question to Gemini: {question}")
            answer = call_gemini(prompt)
            status = "  Not in document" if "not available in the document" in answer.lower() else "✅ Answered"
            print(f"\n{status}")
            print(f" Answer: {answer.strip()}\n")

        except RuntimeError as e:
            logger.error(f"Failed to get answer: {e}")
            print(f" Error: {e}\n")

if __name__ == "__main__":
    main()