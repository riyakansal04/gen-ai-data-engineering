from google import genai
import os
import json
import time
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
# RAG PIPELINE (reused from task8)
# ============================================================
def extract_pdf_text(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += f"\n{page_text}"
    if not text.strip():
        raise ValueError(f"Could not extract text from PDF: {filepath}")
    return text

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def get_embedding(text):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(WAIT_SECONDS * attempt)
            else:
                raise RuntimeError(f"Embedding failed: {e}")

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk))
        time.sleep(1)
    return embeddings

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm != 0 else 0.0

def retrieve_top_k(question, chunks, chunk_embeddings, top_k):
    question_embedding = get_embedding(question)
    scored = [
        (cosine_similarity(question_embedding, emb), i, chunks[i])
        for i, emb in enumerate(chunk_embeddings)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]]

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
                    time.sleep(WAIT_SECONDS * attempt)
                else:
                    raise RuntimeError(f"Gemini unavailable after {MAX_RETRIES} retries")
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise RuntimeError("API quota exhausted")
            else:
                raise RuntimeError(f"Unexpected error: {error_msg}")

def get_rag_answer(question, chunks, chunk_embeddings, rag_template):
    top_chunks = retrieve_top_k(question, chunks, chunk_embeddings, RAG_TOP_K)
    context = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(top_chunks)]
    )
    prompt = rag_template.replace("{context}", context)
    prompt = prompt.replace("{question}", question)
    return call_gemini(prompt)

# ============================================================
# EVALUATION
# ============================================================
def load_eval_questions(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Eval questions not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['questions'])} evaluation questions")
    return data["questions"]

def load_prompt_template(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content

def evaluate_answer(eval_template, question, expected, actual):
    prompt = eval_template.replace("{question}", question)
    prompt = prompt.replace("{expected}", expected)
    prompt = prompt.replace("{actual}", actual)

    raw = call_gemini(prompt)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Could not parse eval response: {e}")
        return {
            "score": 0,
            "verdict": "PARSE_ERROR",
            "reason": f"Could not parse evaluator response: {raw[:100]}"
        }

# ============================================================
# DISPLAY REPORT
# ============================================================
def display_report(results):
    print("   TASK 9 — EVALUATION REPORT")

    for i, r in enumerate(results, 1):
        verdict_icon = {
            "CORRECT":     "✅",
            "PARTIAL":     "⚠️ ",
            "WRONG":       "❌",
            "HALLUCINATED":"🚨",
            "PARSE_ERROR": "❓"
        }.get(r["verdict"], "❓")

        print(f"\nQ{i} [{r['type'].upper()}]: {r['question']}")
        print(f"   Expected  : {r['expected']}")
        print(f"   Actual    : {r['actual'][:120]}...")
        print(f"   Score     : {r['score']}/10")
        print(f"   Verdict   : {verdict_icon} {r['verdict']}")
        print(f"   Reason    : {r['reason']}")
        print("-" * 70)

    # Summary stats
    total        = len(results)
    correct      = sum(1 for r in results if r["verdict"] == "CORRECT")
    partial      = sum(1 for r in results if r["verdict"] == "PARTIAL")
    wrong        = sum(1 for r in results if r["verdict"] == "WRONG")
    hallucinated = sum(1 for r in results if r["verdict"] == "HALLUCINATED")
    avg_score    = sum(r["score"] for r in results) / total

    factual_results = [r for r in results if r["type"] == "factual"]
    halluc_results  = [r for r in results if r["type"] == "hallucination"]

    factual_correct  = sum(1 for r in factual_results if r["verdict"] == "CORRECT")
    halluc_correct   = sum(1 for r in halluc_results  if r["verdict"] == "CORRECT")

    print("   SUMMARY")
    print(f"Total questions     : {total}")
    print(f"Correct          : {correct}")
    print(f"Partial          : {partial}")
    print(f"Wrong            : {wrong}")
    print(f"Hallucinated     : {hallucinated}")
    print(f"Average score       : {avg_score:.1f}/10")
    print(f"\nFactual accuracy    : {factual_correct}/{len(factual_results)}")
    print(f"Hallucination guard : {halluc_correct}/{len(halluc_results)}")

    # Save results to JSON
    output_path = "data/eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "correct": correct,
                "partial": partial,
                "wrong": wrong,
                "hallucinated": hallucinated,
                "avg_score": round(avg_score, 2),
                "factual_accuracy": f"{factual_correct}/{len(factual_results)}",
                "hallucination_guard": f"{halluc_correct}/{len(halluc_results)}"
            },
            "results": results
        }, f, indent=2)
    logger.info(f"Evaluation results saved to: {output_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 50)
    logger.info("TASK 9 - RAG EVALUATION")
    logger.info("=" * 50)

    # Step 1 — Build RAG index
    logger.info("Building RAG index...")
    text = extract_pdf_text(PATHS["sample_pdf"])
    chunks = chunk_text(text, RAG_CHUNK_SIZE)
    logger.info(f"Embedding {len(chunks)} chunks...")
    chunk_embeddings = embed_chunks(chunks)
    logger.info("RAG index ready")

    # Step 2 — Load templates
    rag_template  = load_prompt_template(PATHS["prompt_task8"])
    eval_template = load_prompt_template(PATHS["prompt_task9"])

    # Step 3 — Load eval questions
    eval_questions = load_eval_questions(PATHS["eval_questions"])

    # Step 4 — Run evaluation
    results = []
    for i, item in enumerate(eval_questions, 1):
        question = item["question"]
        expected = item["expected"]
        qtype    = item["type"]

        logger.info(f"Evaluating {i}/{len(eval_questions)}: {question}")

        # Get RAG answer
        actual = get_rag_answer(question, chunks, chunk_embeddings, rag_template)
        logger.info(f"RAG answer: {actual[:80]}...")

        # Evaluate answer
        eval_result = evaluate_answer(eval_template, question, expected, actual)

        results.append({
            "question" : question,
            "expected" : expected,
            "actual"   : actual.strip(),
            "type"     : qtype,
            "score"    : eval_result.get("score", 0),
            "verdict"  : eval_result.get("verdict", "UNKNOWN"),
            "reason"   : eval_result.get("reason", "")
        })

        time.sleep(2)  # avoid rate limits

    # Step 5 — Display report
    display_report(results)
    logger.info("Task 9 completed successfully!")

if __name__ == "__main__":
    main()