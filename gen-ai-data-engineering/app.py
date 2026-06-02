import streamlit as st
import json
import os
import sqlite3
import numpy as np
import PyPDF2
import pandas as pd
import time
from io import StringIO
from dotenv import load_dotenv
from google import genai
import docx

load_dotenv()

from config import (
    GEMINI_API_KEY, MODEL, MAX_RETRIES,
    WAIT_SECONDS, PATHS, DB_PATH,
    RAG_CHUNK_SIZE, RAG_TOP_K
)

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# SHARED HELPERS
# ============================================================
def call_gemini(prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                if attempt < MAX_RETRIES:
                    time.sleep(WAIT_SECONDS * attempt)
                else:
                    st.error("Service temporarily unavailable. Please retry.")
                    return None
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                st.error("API quota exhausted.")
                return None
            else:
                st.error(f"Unexpected error: {error_msg}")
                return None

def load_prompt(filepath):
    if not os.path.exists(filepath):
        st.error(f"Missing config: {filepath}")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_embedding(text):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(model="gemini-embedding-001", contents=text)
            return result.embeddings[0].values
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(WAIT_SECONDS)
            else:
                st.error(f"Embedding failed: {e}")
                return None

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm != 0 else 0.0

def extract_pdf_text(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += f"\n{t}"
    return text

def extract_word_text(filepath):
    document = docx.Document(filepath)
    return "\n".join([p.text for p in document.paragraphs if p.text.strip()])

def extract_document_text(uploaded_file):
    filename = uploaded_file.name.lower()
    temp_path = f"data/temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        if filename.endswith(".pdf"):
            text = extract_pdf_text(temp_path)
        elif filename.endswith((".docx", ".doc")):
            text = extract_word_text(temp_path)
        else:
            return None, "Unsupported file type. Upload PDF or Word."
        if not text.strip():
            return None, "Could not extract text from document."
        return text, None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def get_db_schema():
    """Dynamically read schema from the SQLite database."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        schema_parts = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = cursor.fetchall()
            col_lines = []
            for col in cols:
                cid, name, typ, notnull, dflt, pk = col
                pk_str    = " PRIMARY KEY" if pk else ""
                null_str  = " NOT NULL"    if notnull else ""
                col_lines.append(f"    {name} {typ}{pk_str}{null_str}")
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            fks = cursor.fetchall()
            for fk in fks:
                col_lines.append(f"    FOREIGN KEY({fk[3]}) REFERENCES {fk[2]}({fk[4]})")
            schema_parts.append(f"{table} (\n" + ",\n".join(col_lines) + "\n)")
        conn.close()
        return "\n\n".join(schema_parts)
    except Exception as e:
        return f"-- Could not read schema: {e}"

# ============================================================
# CSS — Sharp Engineering Terminal Aesthetic
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    * { box-sizing: border-box; }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* Background — deep navy, not pure black */
    .stApp { background: #0A0E1A; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #060910 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebarContent"] { padding: 0 !important; }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }

    /* ── TYPOGRAPHY ── */
    h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; }

    /* ── BUTTONS ── */
    .stButton > button {
        background: transparent;
        color: #A8B4C8;
        border: 1px solid rgba(168,180,200,0.18);
        border-radius: 4px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 0.55rem 1.1rem;
        letter-spacing: 0.01em;
        transition: all 0.18s ease;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(255,255,255,0.35);
        color: #FFFFFF;
        transform: none;
        box-shadow: none;
    }
    .stButton > button:active {
        background: rgba(255,255,255,0.07);
    }

    /* Primary run buttons — accent color */
    .run-btn .stButton > button {
        background: #1A56FF;
        border-color: #1A56FF;
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: 0.03em;
        font-size: 0.82rem;
        text-transform: uppercase;
    }
    .run-btn .stButton > button:hover {
        background: #3369FF;
        border-color: #3369FF;
        color: #FFFFFF;
        box-shadow: 0 0 20px rgba(26,86,255,0.35);
    }

    /* ── INPUTS ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #0D1220 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #E2E8F4 !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 0.8rem !important;
        transition: border-color 0.15s !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1A56FF !important;
        box-shadow: 0 0 0 2px rgba(26,86,255,0.15) !important;
        outline: none !important;
    }
    .stTextInput > label, .stTextArea > label {
        display: none !important;
    }

    /* ── METRICS ── */
    [data-testid="stMetric"] {
        background: #0D1220;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 4px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        color: #5A6A82 !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stMetricValue"] {
        color: #E2E8F4 !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #5A6A82 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        padding: 0.65rem 1.4rem !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #E2E8F4 !important;
        border-bottom: 2px solid #1A56FF !important;
        background: transparent !important;
    }

    /* ── DATAFRAME ── */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 4px !important;
    }
    .stDataFrame thead tr th {
        background: #0D1220 !important;
        color: #5A6A82 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    }
    .stDataFrame tbody tr td {
        color: #A8B4C8 !important;
        font-size: 0.88rem !important;
        border-color: rgba(255,255,255,0.04) !important;
    }
    .stDataFrame tbody tr:hover td {
        background: rgba(255,255,255,0.025) !important;
    }

    /* ── CODE BLOCKS ── */
    .stCode, [data-testid="stCode"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    pre {
        background: #060910 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 4px !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: #0D1220 !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
        border-radius: 4px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(26,86,255,0.5) !important;
    }

    /* ── NUMBER INPUT ── */
    .stNumberInput input {
        background: #0D1220 !important;
        border-color: rgba(255,255,255,0.1) !important;
        color: #E2E8F4 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* ── SIDEBAR RADIO ── */
    .stRadio > div { gap: 0 !important; }
    .stRadio label {
        color: #4A5A72 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 1rem !important;
        border-radius: 2px !important;
        margin: 1px 0 !important;
        transition: color 0.12s, background 0.12s !important;
        cursor: pointer !important;
        letter-spacing: 0.01em !important;
    }
    .stRadio label:hover {
        color: #A8B4C8 !important;
        background: rgba(255,255,255,0.04) !important;
    }
    /* Selected radio */
    .stRadio label[data-checked="true"],
    .stRadio [aria-checked="true"] label {
        color: #E2E8F4 !important;
        background: rgba(26,86,255,0.12) !important;
        border-left: 2px solid #1A56FF !important;
    }

    /* ── PROGRESS BAR ── */
    .stProgress > div > div {
        background: #1A56FF !important;
        border-radius: 2px !important;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 2px !important;
    }

    /* ── ALERTS ── */
    .stAlert {
        border-radius: 4px !important;
        border-width: 1px !important;
        font-size: 0.88rem !important;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        background: #0D1220 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 4px !important;
        color: #5A6A82 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    /* ── SPINNER ── */
    .stSpinner > div { border-top-color: #1A56FF !important; }

    /* ── DIVIDER ── */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 1.6rem 0 !important;
    }

    /* ── SELECT ── */
    .stSelectbox > div > div {
        background: #0D1220 !important;
        border-color: rgba(255,255,255,0.1) !important;
        color: #E2E8F4 !important;
    }

    /* ── CUSTOM COMPONENTS ── */

    /* Page header block */
    .pg-header {
        padding: 2.5rem 0 2rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 2rem;
    }
    .pg-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        color: #1A56FF;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .pg-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #E2E8F4;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0 0 0.5rem 0;
    }
    .pg-sub {
        font-size: 0.92rem;
        color: #5A6A82;
        line-height: 1.6;
        max-width: 560px;
        margin: 0;
    }

    /* Section label */
    .sec-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: #3A4A62;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* Stat card */
    .stat-card {
        background: #0D1220;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 4px;
        padding: 1rem 1.2rem;
    }
    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #3A4A62;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #E2E8F4;
        letter-spacing: -0.02em;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stat-sub {
        font-size: 0.72rem;
        color: #3A4A62;
        margin-top: 0.2rem;
    }

    /* Info card */
    .info-card {
        background: #0D1220;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .info-card.accent-l { border-left: 2px solid #1A56FF; }
    .info-card.accent-g { border-left: 2px solid #00C873; }

    /* Answer block */
    .answer-block {
        background: #0D1220;
        border: 1px solid rgba(255,255,255,0.07);
        border-left: 2px solid #00C873;
        border-radius: 0 4px 4px 0;
        padding: 1.2rem 1.4rem;
        font-size: 0.92rem;
        color: #C8D4E8;
        line-height: 1.75;
    }

    /* Row item */
    .row-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0D1220;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 3px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.35rem;
        font-size: 0.88rem;
        color: #A8B4C8;
    }
    .row-item .name { color: #E2E8F4; font-weight: 500; }
    .row-item .amount-pos {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #00C873;
        font-weight: 500;
    }
    .row-item .amount-zero {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #3A4A62;
    }

    /* Home tool card */
    .tool-card {
        background: #0D1220;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 4px;
        padding: 1.2rem 1.4rem;
        cursor: pointer;
        transition: border-color 0.15s, background 0.15s;
        height: 100%;
    }
    .tool-card:hover {
        background: #111827;
        border-color: rgba(26,86,255,0.5);
    }
    .tool-card-name {
        font-size: 0.92rem;
        font-weight: 600;
        color: #E2E8F4;
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }
    .tool-card-desc {
        font-size: 0.8rem;
        color: #3A4A62;
        line-height: 1.55;
    }

    /* Status dot */
    .status-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-right: 0.4rem;
        vertical-align: middle;
    }
    .dot-green { background: #00C873; box-shadow: 0 0 6px #00C873; }
    .dot-red   { background: #FF3B3B; }

    /* Similarity score */
    .sim-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.5rem;
        font-weight: 600;
        letter-spacing: -0.04em;
    }

    /* Rank row */
    .rank-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0D1220;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 3px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.4rem;
    }
    .rank-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #3A4A62;
        margin-right: 0.8rem;
        min-width: 24px;
    }
    .rank-text { color: #A8B4C8; font-size: 0.88rem; flex: 1; }
    .rank-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #5A6A82;
        margin-left: 1rem;
    }

    /* Sidebar nav item active indicator */
    div[data-testid="stSidebar"] .stRadio > div > label[data-baseweb] {
        border-left: 2px solid transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

def label(text):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)

def pg_header(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="pg-header">
        <div class="pg-eyebrow">{eyebrow}</div>
        <div class="pg-title">{title}</div>
        <div class="pg-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: HOME
# ============================================================
def page_home():
    st.markdown("""
    <div style="padding: 3.5rem 0 2rem 0">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#1A56FF;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.2rem">
            DataForge AI &nbsp;/&nbsp; Data Engineering Platform
        </div>
        <h1 style="font-size:3rem; font-weight:700; color:#E2E8F4; margin:0 0 0.9rem 0;
                   letter-spacing:-0.04em; line-height:1.05">
            AI Tools for<br>Data Engineers
        </h1>
        <p style="font-size:1rem; color:#3A4A62; line-height:1.75; max-width:520px; margin:0">
            Query databases in plain English. Analyze documents.
            Generate synthetic data. Explore vector embeddings.
            All powered by Gemini.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    tools = [
        ("SQL Optimizer",        "Rewrite slow queries, generate data quality checks, and design scalable pipeline architectures.",  "SQL Optimizer"),
        ("Activity Analyzer",    "Parse any user activity log and extract structured business metrics and insights.",                  "Activity Analyzer"),
        ("Test Data Generator",  "Generate statistically consistent synthetic rows from an existing CSV schema.",                     "Test Data Generator"),
        ("Document Intelligence","Ask natural language questions against any PDF or Word document.",                                   "Document Intelligence"),
        ("NL Query Engine",      "Type a question in plain English — get back a SQL query and live results.",                         "NL Query Engine"),
        ("Semantic Search",      "Measure cosine similarity between texts and rank documents by semantic relevance.",                  "Semantic Search"),
        ("Knowledge Base Q&A",   "RAG pipeline: chunk, embed, retrieve, and answer questions grounded in your documents.",            "Knowledge Base Q&A"),
    ]

    pages_map = {
        "SQL Optimizer":        "SQL Optimizer",
        "Activity Analyzer":    "Activity Analyzer",
        "Test Data Generator":  "Test Data Generator",
        "Document Intelligence":"Document Intelligence",
        "NL Query Engine":      "NL Query Engine",
        "Semantic Search":      "Semantic Search",
        "Knowledge Base Q&A":   "Knowledge Base Q&A",
    }

    cols = st.columns(2, gap="small")
    for i, (name, desc, nav) in enumerate(tools):
        with cols[i % 2]:
            if st.button(
                f"{name}\n\n{desc}",
                key=f"home_{name}",
                use_container_width=True
            ):
                st.session_state["nav_page"] = nav
                st.rerun()

    st.divider()

    checks = [
        ("Gemini API",  bool(GEMINI_API_KEY)),
        ("Sales PDF",   os.path.exists(PATHS.get("sample_pdf", ""))),
        ("Database",    os.path.exists(DB_PATH)),
        ("Sample CSV",  os.path.exists(PATHS.get("sample_csv", ""))),
    ]
    c = st.columns(len(checks))
    for col, (lbl, ok) in zip(c, checks):
        dot   = "dot-green" if ok else "dot-red"
        color = "#00C873"   if ok else "#FF3B3B"
        text  = "Ready"     if ok else "Missing"
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{lbl}</div>
            <div style="display:flex;align-items:center;margin-top:0.3rem">
                <span class="status-dot {dot}"></span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.88rem;
                             color:{color};font-weight:500">{text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE: SQL OPTIMIZER
# ============================================================
def page_sql_optimizer():
    pg_header("Tool — 01", "SQL Optimizer", "Query optimization, data quality checks, and pipeline design — AI-assisted.")

    tab1, tab2, tab3 = st.tabs(["Query Optimizer", "Data Quality", "Pipeline Design"])

    with tab1:
        label("SQL Query")
        query = st.text_area("q", height=160,
            placeholder="SELECT * FROM orders o\nJOIN customers c ON o.customer_id = c.id\nWHERE o.created_at > '2024-01-01'",
            label_visibility="collapsed")
        label("Context (optional)")
        ctx = st.text_input("ctx",
            placeholder="Database: Snowflake · Scale: 500M rows · Indexes: none on date columns",
            label_visibility="collapsed")
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run = st.button("Optimize", key="opt_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if run:
            if query.strip():
                p = load_prompt(PATHS["prompt_task1_1"])
                if p:
                    with st.spinner("Analyzing..."):
                        r = call_gemini(f"{p}\n\nQuery:\n{query}\nContext: {ctx}")
                    if r:
                        st.divider()
                        st.markdown(r)
            else:
                st.warning("Enter a SQL query.")

    with tab2:
        label("Dataset Description")
        ds = st.text_area("ds", height=120,
            placeholder="Customer transactions table · 10M rows · Columns: customer_id, amount, date, status",
            label_visibility="collapsed")
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run2 = st.button("Generate Checks", key="dq_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if run2:
            if ds.strip():
                p = load_prompt(PATHS["prompt_task1_2"])
                if p:
                    with st.spinner("Generating..."):
                        r = call_gemini(f"{p}\n\nDataset: {ds}")
                    if r:
                        st.divider()
                        st.markdown(r)
            else:
                st.warning("Describe your dataset.")

    with tab3:
        label("Pipeline Requirements")
        req = st.text_area("req", height=120,
            placeholder="Ingest 1M events/day from Kafka · Transform and load to Snowflake · Latency < 5 min",
            label_visibility="collapsed")
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run3 = st.button("Design Pipeline", key="pipe_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if run3:
            if req.strip():
                p = load_prompt(PATHS["prompt_task1_3"])
                if p:
                    with st.spinner("Designing..."):
                        r = call_gemini(f"{p}\n\nRequirements: {req}")
                    if r:
                        st.divider()
                        st.markdown(r)
            else:
                st.warning("Describe your requirements.")

# ============================================================
# PAGE: ACTIVITY ANALYZER
# ============================================================
def page_activity_analyzer():
    pg_header("Tool — 02", "Activity Analyzer", "Paste any user activity log and extract structured metrics instantly.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        label("Input Log  ·  Format: User | action | amount")
        default = "User A | logged in and purchased a laptop | 1200\nUser B | logged in but did not make any purchase | 0\nUser C | purchased a phone | 800"
        activity_input = st.text_area("log", value=default, height=200,
            placeholder="User A | purchased laptop | 1200\nUser B | browsed only | 0",
            label_visibility="collapsed")
        st.markdown('<div style="font-size:0.75rem;color:#3A4A62;margin-top:0.3rem;margin-bottom:1rem">One entry per line. Amount column is optional (use 0 for no purchase).</div>', unsafe_allow_html=True)
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run = st.button("Run Analysis", key="act_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run:
            lines = [l.strip() for l in activity_input.strip().split("\n") if l.strip()]
            activities, errors = [], []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2:
                    errors.append(f"Line {i+1}: missing fields — expected User | action | amount")
                    continue
                try:
                    amount = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0
                except ValueError:
                    errors.append(f"Line {i+1}: invalid amount '{parts[2]}', using 0")
                    amount = 0.0
                activities.append({"user": parts[0], "action": parts[1], "amount": amount})

            for e in errors:
                st.warning(e)

            if not activities:
                st.error("No valid entries. Check format.")
            else:
                tmpl = load_prompt(PATHS["prompt_task2"])
                if tmpl:
                    txt = "\n".join([
                        f"- {a['user']} {a['action']}" + (f" worth ${a['amount']}" if a["amount"] > 0 else "")
                        for a in activities
                    ])
                    prompt = tmpl.replace("{activity_text}", txt).replace("{total_users}", str(len(activities)))
                    with st.spinner("Analyzing..."):
                        raw = call_gemini(prompt)
                    if raw:
                        try:
                            cleaned = raw.strip()
                            if cleaned.startswith("```"):
                                cleaned = cleaned.split("```")[1]
                                if cleaned.startswith("json"):
                                    cleaned = cleaned[4:]
                            result = json.loads(cleaned.strip())
                            result["_activities"] = activities
                            st.session_state["act_result"] = result
                        except Exception as e:
                            st.error(f"Parse error: {e}")

    with col2:
        label("Analysis Output")
        if "act_result" in st.session_state:
            r = st.session_state["act_result"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Users",    r.get("total_users", "—"))
            c2.metric("Buyers",   r.get("purchasing_users", "—"))
            c3.metric("Revenue",  f"${r.get('total_revenue', 0):,}")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-card accent-l">
                <div class="sec-label" style="margin-bottom:0.4rem">Summary</div>
                <div style="font-size:0.9rem;color:#A8B4C8;line-height:1.65">{r.get("summary","")}</div>
            </div>
            """, unsafe_allow_html=True)

            label("Key Insights")
            for ins in r.get("insights", []):
                st.markdown(f"""
                <div class="row-item">
                    <span style="color:#1A56FF;font-family:'IBM Plex Mono',monospace;
                                 font-size:0.7rem;margin-right:0.6rem;min-width:12px">→</span>
                    <span>{ins}</span>
                </div>
                """, unsafe_allow_html=True)

            if "_activities" in r:
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                label("Breakdown")
                for a in r["_activities"]:
                    amt_html = (
                        f'<span class="amount-pos">+${a["amount"]:,.0f}</span>'
                        if a["amount"] > 0
                        else '<span class="amount-zero">—</span>'
                    )
                    st.markdown(f"""
                    <div class="row-item">
                        <span><span class="name">{a['user']}</span>
                        &nbsp;<span style="color:#3A4A62">·</span>&nbsp;
                        {a['action']}</span>
                        {amt_html}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding:4rem 0;text-align:center;color:#3A4A62">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                            letter-spacing:0.08em;text-transform:uppercase">
                    Awaiting input
                </div>
                <div style="font-size:0.82rem;margin-top:0.5rem">
                    Edit the log and run analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: TEST DATA GENERATOR
# ============================================================
def page_data_generator():
    pg_header("Tool — 03", "Test Data Generator", "Generate statistically consistent synthetic rows from your CSV schema.")

    if not os.path.exists(PATHS["sample_csv"]):
        st.error(f"sample.csv not found at: {PATHS['sample_csv']}")
        return

    orig = pd.read_csv(PATHS["sample_csv"])
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        label("Configuration")
        num_rows = st.number_input("Rows to generate", min_value=1, max_value=100, value=10)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        label("Source Schema")
        schema_df = orig.dtypes.rename("type").reset_index()
        schema_df.columns = ["column", "dtype"]
        st.dataframe(schema_df, hide_index=True, use_container_width=True)

        st.markdown(f"""
        <div class="info-card" style="margin-top:0.5rem;font-size:0.8rem">
            <span style="color:#E2E8F4;font-family:'IBM Plex Mono',monospace">{len(orig)}</span>
            <span style="color:#3A4A62"> source rows &nbsp;·&nbsp; </span>
            <span style="color:#E2E8F4;font-family:'IBM Plex Mono',monospace">{len(orig.columns)}</span>
            <span style="color:#3A4A62"> columns</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run = st.button("Generate", key="gen_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run:
            tmpl = load_prompt(PATHS["prompt_task3"])
            if tmpl:
                start_id = orig["employee_id"].max() + 1
                prompt = (tmpl
                    .replace("{csv_content}", orig.to_csv(index=False))
                    .replace("{num_rows}", str(num_rows))
                    .replace("{start_id}", str(start_id)))
                with st.spinner(f"Generating {num_rows} rows..."):
                    raw = call_gemini(prompt)
                if raw:
                    try:
                        cleaned = raw.strip()
                        if cleaned.startswith("```"):
                            cleaned = cleaned.split("```")[1]
                            if cleaned.startswith("csv"):
                                cleaned = cleaned[3:]
                        new_df = pd.read_csv(StringIO(cleaned.strip()), header=None, names=orig.columns)
                        combined = pd.concat([orig, new_df], ignore_index=True)
                        st.session_state["gen_new"] = new_df
                        st.session_state["gen_combined"] = combined
                        st.success(f"Generated {len(new_df)} rows")
                    except Exception as e:
                        st.error(f"Parse error: {e}")

    with col2:
        label("Output")
        if "gen_new" in st.session_state:
            combined = st.session_state["gen_combined"]
            new_df   = st.session_state["gen_new"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows",   len(combined))
            c2.metric("Avg Salary",   f"${combined['salary'].mean():,.0f}")
            c3.metric("Departments",  combined["department"].nunique())

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Generated", "Full Dataset"])
            with tab1:
                st.dataframe(new_df, use_container_width=True, hide_index=True)
            with tab2:
                st.dataframe(combined, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="padding:4rem 0;text-align:center;color:#3A4A62">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                            letter-spacing:0.08em;text-transform:uppercase">Awaiting generation</div>
                <div style="font-size:0.82rem;margin-top:0.5rem">Configure and click Generate</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: DOCUMENT INTELLIGENCE
# ============================================================
def page_document_qa():
    pg_header("Tool — 04", "Document Intelligence", "Ask natural language questions against any PDF or Word document.")

    tmpl = load_prompt(PATHS["prompt_task4"])
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        label("Document Source")
        uploaded = st.file_uploader("Upload PDF or Word", type=["pdf","docx","doc"],
                                    label_visibility="collapsed")

        if not uploaded:
            if os.path.exists(PATHS.get("sample_pdf", "")):
                st.info("Using default: Acme Corp Sales Report 2024")
                doc_text = extract_pdf_text(PATHS["sample_pdf"])
                doc_name = "sample.pdf"
            else:
                st.warning("Upload a document to begin.")
                return
        else:
            with st.spinner("Extracting..."):
                doc_text, err = extract_document_text(uploaded)
            if err:
                st.error(err)
                return
            doc_name = uploaded.name

        st.markdown(f"""
        <div class="info-card" style="margin:0.6rem 0">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#1A56FF">{doc_name}</div>
            <div style="font-size:0.72rem;color:#3A4A62;margin-top:0.2rem">{len(doc_text):,} chars extracted</div>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic AI-generated suggestions
        doc_key = f"sugg_{doc_name}"
        if doc_key not in st.session_state:
            with st.spinner("Generating suggestions..."):
                sp = f"""Generate 4 short, specific questions a user might ask about this document.
Return ONLY a JSON array of 4 strings. No markdown, no explanation.
Document excerpt: {doc_text[:2000]}"""
                raw = call_gemini(sp)
                try:
                    cleaned = raw.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("```")[1]
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:]
                    suggs = json.loads(cleaned.strip())
                    if not isinstance(suggs, list):
                        raise ValueError
                except Exception:
                    suggs = ["Summarize the document",
                             "What are the key findings?",
                             "What risks are mentioned?",
                             "What are the recommendations?"]
            st.session_state[doc_key] = suggs

        label("Suggested Questions")
        for s in st.session_state[doc_key]:
            if st.button(s, key=f"s_{hash(s)}", use_container_width=True):
                st.session_state["doc_q"] = s

    with col2:
        label("Ask a Question")
        q = st.text_input("q", value=st.session_state.get("doc_q", ""),
            placeholder="What was the total revenue in 2024?",
            label_visibility="collapsed")
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run = st.button("Get Answer", key="doc_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run and q:
            prompt = tmpl.replace("{document_content}", doc_text).replace("{question}", q)
            with st.spinner("Searching document..."):
                ans = call_gemini(prompt)
            if ans:
                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                label("Answer")
                if "not available" in ans.lower():
                    st.warning(ans.strip())
                else:
                    st.markdown(f'<div class="answer-block">{ans.strip().replace(chr(10), "<br>")}</div>',
                                unsafe_allow_html=True)

# ============================================================
# PAGE: NL QUERY ENGINE
# ============================================================
def page_nl_query():
    pg_header("Tool — 05", "NL Query Engine", "Type a question in plain English — get back SQL and live results.")

    if not os.path.exists(DB_PATH):
        st.warning(f"Database not found at `{DB_PATH}`. Run task5_texttosql.py first.")
        return

    schema = get_db_schema()
    tmpl   = load_prompt(PATHS["prompt_task5"])

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        label("Live Database Schema")
        st.code(schema or "-- Schema unavailable", language="sql")

    with col2:
        label("Your Question")
        q = st.text_input("q",
            value=st.session_state.get("nl_q", ""),
            placeholder="Which customers spent the most in the last 7 days?",
            label_visibility="collapsed")
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run = st.button("Run Query", key="nl_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run and q:
            prompt = tmpl.replace("{schema}", schema).replace("{question}", q)
            with st.spinner("Generating SQL..."):
                raw_sql = call_gemini(prompt)
            if raw_sql:
                cleaned = raw_sql.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.lower().startswith("sql"):
                        cleaned = cleaned[3:]
                sql = cleaned.strip()

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                label("Generated SQL")
                st.code(sql, language="sql")

                try:
                    conn = sqlite3.connect(DB_PATH)
                    df   = pd.read_sql_query(sql, conn)
                    conn.close()
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    label(f"Results  ·  {len(df)} row(s)")
                    if df.empty:
                        st.info("No results returned.")
                    else:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"SQL execution error: {e}")

# ============================================================
# PAGE: SEMANTIC SEARCH
# ============================================================
def page_semantic_search():
    pg_header("Tool — 06", "Semantic Search", "Measure text similarity and rank documents by semantic relevance.")

    tab1, tab2 = st.tabs(["Similarity Explorer", "Document Ranker"])

    with tab1:
        st.markdown('<div style="font-size:0.88rem;color:#3A4A62;margin-bottom:1.2rem;line-height:1.6">Compare two texts and compute their cosine similarity using Gemini embeddings.</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            label("Text A")
            t1 = st.text_area("t1", "Revenue increased by 12% in fiscal year 2024", height=110,
                              label_visibility="collapsed")
        with col2:
            label("Text B")
            t2 = st.text_area("t2", "Annual sales grew by 12 percent this year", height=110,
                              label_visibility="collapsed")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run = st.button("Compute Similarity", key="sim_run", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        if run:
            with st.spinner("Computing embeddings..."):
                e1 = get_embedding(t1)
                e2 = get_embedding(t2)
            if e1 and e2:
                score = cosine_similarity(e1, e2)
                interp = "VERY SIMILAR" if score > 0.85 else "RELATED" if score > 0.65 else "DIFFERENT"
                color  = "#00C873"       if score > 0.85 else "#F5A623"  if score > 0.65 else "#FF3B3B"

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div class="stat-card" style="text-align:center;padding:1.8rem 1rem">
                        <div class="stat-label">Cosine Similarity</div>
                        <div class="sim-score" style="color:{color}">{score:.4f}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                                    color:{color};letter-spacing:0.1em;margin-top:0.4rem">{interp}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.progress(float(score))
                    c3, c4 = st.columns(2)
                    c3.metric("Dimensions", len(e1))
                    c4.metric("Score",      f"{score:.4f}")
                    with st.expander("Raw embedding preview (first 10 dims)"):
                        st.write("A:", [round(x, 4) for x in e1[:10]])
                        st.write("B:", [round(x, 4) for x in e2[:10]])

    with tab2:
        st.markdown('<div style="font-size:0.88rem;color:#3A4A62;margin-bottom:1.2rem;line-height:1.6">Rank a set of documents by their semantic similarity to a search query.</div>',
                    unsafe_allow_html=True)
        label("Search Query")
        query = st.text_input("qr",
            "Which product had the highest revenue?",
            label_visibility="collapsed")
        label("Documents  ·  One per line")
        cands_raw = st.text_area("cands",
            "Cloud Storage Pro generated $12.5M in revenue\nLaptop was the most sold product\nNorth America led regional revenue at $22.1M\nCustomer churn increased in SMB segment",
            height=120, label_visibility="collapsed")

        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run2 = st.button("Rank Documents", key="rank_run", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        if run2:
            cands = [c.strip() for c in cands_raw.split("\n") if c.strip()]
            with st.spinner("Computing similarities..."):
                qe  = get_embedding(query)
                scored = []
                for c in cands:
                    emb = get_embedding(c)
                    if emb:
                        scored.append((cosine_similarity(qe, emb), c))
                    time.sleep(0.3)
            scored.sort(reverse=True)

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            label("Ranked Results")
            for rank, (score, text) in enumerate(scored, 1):
                rank_color = "#00C873" if rank == 1 else "#1A56FF" if rank == 2 else "#3A4A62"
                st.markdown(f"""
                <div class="rank-row">
                    <span class="rank-num" style="color:{rank_color}">#{rank:02d}</span>
                    <span class="rank-text">{text}</span>
                    <span class="rank-score">{score:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(score))

# ============================================================
# PAGE: KNOWLEDGE BASE Q&A
# ============================================================
def page_knowledge_base():
    pg_header("Tool — 07", "Knowledge Base Q&A", "RAG pipeline: chunk, embed, retrieve, and answer from your documents.")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        label("Knowledge Source")
        uploaded = st.file_uploader("Upload document", type=["pdf","docx","doc"],
                                    key="rag_up", label_visibility="collapsed")

        if not uploaded:
            if not os.path.exists(PATHS.get("sample_pdf", "")):
                st.error("No document found.")
                return
            src      = PATHS["sample_pdf"]
            doc_name = "Acme Corp Sales Report 2024"
            doc_key  = "sample"
        else:
            src      = uploaded
            doc_name = uploaded.name
            doc_key  = uploaded.name

        if st.session_state.get("rag_key") != doc_key:
            st.session_state.pop("rag_chunks", None)
            st.session_state.pop("rag_embs",   None)
            st.session_state["rag_key"] = doc_key

        if "rag_chunks" not in st.session_state:
            with st.spinner("Indexing document..."):
                text = extract_pdf_text(src) if isinstance(src, str) else extract_document_text(src)[0]
                chunks = chunk_text(text, RAG_CHUNK_SIZE)
                embs   = []
                for ch in chunks:
                    embs.append(get_embedding(ch))
                    time.sleep(1)
                st.session_state["rag_chunks"] = chunks
                st.session_state["rag_embs"]   = embs

        chunks = st.session_state["rag_chunks"]
        embs   = st.session_state["rag_embs"]

        st.markdown(f"""
        <div class="info-card accent-g" style="margin:0.6rem 0">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#00C873;
                        letter-spacing:0.1em;margin-bottom:0.3rem">INDEXED</div>
            <div style="font-size:0.88rem;font-weight:600;color:#E2E8F4">{doc_name}</div>
            <div style="font-size:0.72rem;color:#3A4A62;margin-top:0.2rem">
                {len(chunks)} chunks &nbsp;·&nbsp; top-{RAG_TOP_K} retrieval
            </div>
        </div>
        """, unsafe_allow_html=True)

        label("Quick Questions")
        for q in ["What was total revenue?", "What are the 2025 goals?", "Any supply chain issues?"]:
            if st.button(q, key=f"rag_{hash(q)}", use_container_width=True):
                st.session_state["rag_q"] = q

    with col2:
        tmpl = load_prompt(PATHS.get("prompt_task8", "prompts/prompt_task8.txt"))
        label("Your Question")
        question = st.text_input("rq", value=st.session_state.get("rag_q", ""),
            placeholder="What were the key challenges in 2024?",
            label_visibility="collapsed")
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run = st.button("Search & Answer", key="rag_run", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run and question:
            with st.spinner("Retrieving context..."):
                qe = get_embedding(question)
                scored = sorted([
                    (cosine_similarity(qe, emb), i, chunks[i])
                    for i, emb in enumerate(embs)
                ], reverse=True)
                top = scored[:RAG_TOP_K]

            with st.expander(f"Retrieved chunks ({len(top)})"):
                for score, idx, chunk in top:
                    st.markdown(f"**Chunk {idx+1}** — `{score:.4f}`")
                    st.text(chunk[:200] + "…")

            context = "\n\n---\n\n".join([f"[Chunk {i+1}]\n{c}" for _, i, c in top])
            if tmpl:
                prompt = tmpl.replace("{context}", context).replace("{question}", question)
                with st.spinner("Generating answer..."):
                    ans = call_gemini(prompt)
                if ans:
                    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                    label("Answer")
                    if "not available" in ans.lower():
                        st.warning(ans.strip())
                    else:
                        st.markdown(f'<div class="answer-block">{ans.strip().replace(chr(10), "<br>")}</div>',
                                    unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="DataForge AI",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    page_labels = {
        "Home":                 page_home,
        "SQL Optimizer":        page_sql_optimizer,
        "Activity Analyzer":    page_activity_analyzer,
        "Test Data Generator":  page_data_generator,
        "Document Intelligence":page_document_qa,
        "NL Query Engine":      page_nl_query,
        "Semantic Search":      page_semantic_search,
        "Knowledge Base Q&A":   page_knowledge_base,
    }
    keys = list(page_labels.keys())

    # Handle home card navigation
    if "nav_page" in st.session_state:
        nav = st.session_state.pop("nav_page")
        default_idx = keys.index(nav) if nav in keys else 0
    else:
        default_idx = 0

    with st.sidebar:
        st.markdown("""
        <div style="padding:1.8rem 1rem 0.5rem 1rem">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#1A56FF;
                        letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem">
                DataForge
            </div>
            <div style="font-size:1rem;font-weight:700;color:#E2E8F4;letter-spacing:-0.02em">
                AI Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        selected = st.radio("nav", keys, index=default_idx, label_visibility="collapsed")

        st.divider()

        st.markdown(f"""
        <div style="padding:0 1rem 1rem 1rem;font-family:'IBM Plex Mono',monospace;
                    font-size:0.65rem;color:#3A4A62;line-height:2.2">
            MODEL<br>
            <span style="color:#5A6A82">{MODEL}</span><br><br>
            ENGINE<br>
            <span style="color:#5A6A82">Google Gemini</span>
        </div>
        """, unsafe_allow_html=True)

    page_labels[selected]()

if __name__ == "__main__":
    main()