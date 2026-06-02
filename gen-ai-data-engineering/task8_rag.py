from google import genai
import os
import time
import json
import logging
import numpy as np
import PyPDF2
from dotenv import load_dotenv
from config import (
    GEMINI_API_KEY, MODEL, MAX_RETRIES,
    WAIT_SECONDS, PATHS, RAG_CHUNK_SIZE, RAG_TOP_K
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
# STEP 1 — Extract text from PDF
# ============================================================
def extract_pdf_text(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n{page_text}"
    if not text.strip():
        raise ValueError(f"Could not extract text from PDF: {filepath}")
    logger.info(f"Extracted {len(text)} characters from PDF")
    return text

# ============================================================
# STEP 2 — Chunk text
# ============================================================
def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks of ~{chunk_size} words each")
    return chunks

# ============================================================
# STEP 3 — Get embedding from Gemini
# ============================================================
def get_embedding(text, retries=3):
    for attempt in range(1, retries + 1):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            error_msg = str(e)
            if attempt < retries:
                wait = WAIT_SECONDS * attempt
                logger.warning(f"Embedding attempt {attempt} failed: {error_msg}")
                logger.warning(f"Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Embedding failed after {retries} retries: {error_msg}")

# ============================================================
# STEP 4 — Embed all chunks
# ============================================================
def embed_chunks(chunks):
    logger.info(f"Embedding {len(chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        logger.info(f"Embedded chunk {i+1}/{len(chunks)}")
        time.sleep(1)  # avoid rate limits
    logger.info("All chunks embedded successfully")
    return embeddings

# ============================================================
# STEP 5 — Cosine similarity
# ============================================================
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(dot / norm)

# ============================================================
# STEP 6 — Retrieve top K most relevant chunks
# ============================================================
def retrieve_top_k(question, chunks, chunk_embeddings, top_k):
    logger.info(f"Embedding question and retrieving top {top_k} chunks...")
    question_embedding = get_embedding(question)

    scored = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_emb)
        scored.append((score, i, chunks[i]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    logger.info("Top chunks retrieved:")
    for rank, (score, idx, _) in enumerate(top, 1):
        logger.info(f"  Rank {rank} — Chunk {idx+1} — Similarity: {score:.4f}")

    return [chunk for _, _, chunk in top]

# ============================================================
# STEP 7 — Load prompt and build RAG prompt
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

def build_rag_prompt(template, context_chunks, question):
    context = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    prompt = template.replace("{context}", context)
    prompt = prompt.replace("{question}", question)
    return prompt

# ============================================================
# STEP 8 — Call Gemini for final answer
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
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Gemini unavailable after {MAX_RETRIES} retries")
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise RuntimeError("API quota exhausted")
            else:
                raise RuntimeError(f"Unexpected error: {error_msg}")

# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("TASK 8 - RAG SYSTEM")

    # Step 1 — Extract PDF text
    text = extract_pdf_text(PATHS["sample_pdf"])

    # Step 2 — Chunk text
    chunks = chunk_text(text, RAG_CHUNK_SIZE)

    # Step 3 — Embed all chunks (this is the "indexing" phase)
    print("\nIndexing document chunks... (this may take a moment)")
    chunk_embeddings = embed_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks successfully!\n")

    # Step 4 — Load prompt template
    template = load_prompt_template(PATHS["prompt_task8"])

    print("   RAG SYSTEM - RETRIEVAL AUGMENTED GENERATION")
    print(f"   Document: {PATHS['sample_pdf']}")
    print(f"   Chunks: {len(chunks)} | Top-K: {RAG_TOP_K}")
    print("   Type 'exit' to quit")
    input("\nPress Enter to start asking questions...\n")

    # Step 5 — Interactive Q&A loop
    while True:
        question = input("Your question: ").strip()

        if not question:
            print(" Please enter a question\n")
            continue

        if question.lower() == "exit":
            print("\n Exiting RAG system. Goodbye!")
            logger.info("User exited RAG session")
            break

        try:
            # Retrieve relevant chunks
            top_chunks = retrieve_top_k(
                question, chunks, chunk_embeddings, RAG_TOP_K
            )

            # Build RAG prompt
            prompt = build_rag_prompt(template, top_chunks, question)

            # Get answer from Gemini
            logger.info("Sending RAG prompt to Gemini...")
            answer = call_gemini(prompt)

            # Display
            print(f"Question : {question}")
            print(f"Chunks used : {RAG_TOP_K} most relevant")
            print(f"Answer: {answer.strip()}")

        except RuntimeError as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()