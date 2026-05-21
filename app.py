"""
BioASQ RAG Demo — Streamlit app
Pipeline:
    user question
        -> S-PubMedBert encoder embeds the query
        -> FAISS top-k search over biomedical passages
        -> LLaMA-3.3-70B via Groq generates a grounded answer
        -> answer + retrieved passages shown in the UI
"""

import os
import re

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "data"
PASSAGES_PATH = os.path.join(DATA_DIR, "bioasq_passages.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "bioasq_faiss.index")

EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_MODEL = "llama-3.3-70b-versatile"

DEFAULT_TOP_K = 5
MAX_GENERATION_TOKENS = 220
GENERATION_TEMPERATURE = 0.0

PASSAGE_ID_COL = "id"
PASSAGE_TEXT_COL = "passage"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BioASQ RAG Demo",
    page_icon="🧬",
    layout="centered",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main {
        max-width: 980px;
        margin: 0 auto;
    }

    .hero {
        text-align: center;
        padding: 24px 10px 8px;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #14b8a6 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.5;
    }

    .pipeline {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin: 18px 0 20px;
        font-size: 0.85rem;
        color: #475569;
    }

    .pipeline-step {
        padding: 6px 12px;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
    }

    .pipeline-arrow {
        align-self: center;
        opacity: 0.6;
    }

    .disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 16px 0 24px;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .answer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .answer-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #14b8a6;
        margin-bottom: 8px;
    }

    .answer-body {
        font-size: 1.05rem;
        line-height: 1.65;
        color: #0f172a;
    }

    .passage-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    .passage-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #64748b;
        flex-wrap: wrap;
    }

    .passage-rank {
        background: linear-gradient(135deg, #14b8a6 0%, #6366f1 100%);
        color: white;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
    }

    .passage-body {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #1e293b;
    }

    .footer {
        margin-top: 32px;
        padding: 20px 0;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# STARTUP — load everything once
# =========================================================

@st.cache_resource
def load_encoder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH)


@st.cache_data
def load_passages():
    return pd.read_csv(PASSAGES_PATH).reset_index(drop=True)


@st.cache_resource
def load_groq_client():
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        # For Streamlit Cloud, this allows using st.secrets instead of env variables.
        groq_api_key = st.secrets.get("GROQ_API_KEY", None)

    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it as an environment variable "
            "or in Streamlit secrets."
        )

    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )


try:
    passages_df = load_passages()
    index = load_faiss_index()
    encoder = load_encoder()
    groq_client = load_groq_client()

    if len(passages_df) != index.ntotal:
        st.error(
            f"Data mismatch: {len(passages_df)} passages vs {index.ntotal} FAISS vectors."
        )
        st.stop()

except Exception as exc:
    st.error(f"Startup error: {type(exc).__name__}: {exc}")
    st.stop()


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(query: str, k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """Embed the query, search FAISS, and return top-k passages."""
    query_embedding = encoder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    retrieved = passages_df.iloc[indices[0]].copy()
    retrieved["retrieval_score"] = scores[0]
    retrieved["rank"] = range(1, len(retrieved) + 1)

    return retrieved[
        [PASSAGE_ID_COL, PASSAGE_TEXT_COL, "retrieval_score", "rank"]
    ].reset_index(drop=True)


# =========================================================
# GENERATION
# =========================================================

def format_evidence(retrieved_passages: pd.DataFrame) -> str:
    """Format retrieved passages as evidence for the LLM."""
    blocks = []

    for _, row in retrieved_passages.iterrows():
        block = (
            f"Evidence {int(row['rank'])}\n"
            f"Passage ID: {row[PASSAGE_ID_COL]}\n"
            f"Retrieval Score: {float(row['retrieval_score']):.4f}\n"
            f"Text:\n{str(row[PASSAGE_TEXT_COL]).strip()}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks) if blocks else "No usable retrieved evidence."


def build_prompt(question: str, retrieved_passages: pd.DataFrame) -> str:
    """Strict biomedical RAG prompt."""
    evidence_text = format_evidence(retrieved_passages)

    return (
        "You are a biomedical evidence-grounded question-answering assistant.\n\n"
        "Answer the user's biomedical question using ONLY the retrieved evidence.\n\n"
        "Strict rules:\n"
        "1. Use only the retrieved evidence.\n"
        "2. Do not use outside medical, biological, or scientific knowledge.\n"
        "3. Do not invent genes, proteins, diseases, drugs, mechanisms, results, or conclusions.\n"
        "4. Do not mention that you are an AI model.\n"
        "5. Do not include reasoning steps, hidden reasoning, analysis, or <think> tags.\n"
        "6. Return only the final answer.\n"
        "7. Keep the answer concise: maximum 2 sentences.\n"
        "8. If the retrieved evidence directly answers the question, answer it clearly.\n"
        "9. If the retrieved evidence does not contain enough information, write exactly:\n"
        "   The retrieved evidence is insufficient to answer the question.\n"
        "10. Do not cite passage IDs in the final answer unless the user explicitly asks for sources.\n\n"
        f"User Question:\n{question.strip()}\n\n"
        f"Retrieved Evidence:\n{evidence_text}\n\n"
        "Final Answer:"
    )


def clean_answer(text: str) -> str:
    """Clean model output."""
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*(final answer|answer|generated answer)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_answer(prompt: str) -> str:
    """Call Groq-hosted LLaMA and clean the response."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict biomedical evidence-grounded QA assistant. "
                    "You must answer only from the retrieved evidence. "
                    "Return only the final answer without reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MAX_GENERATION_TOKENS,
        top_p=1,
    )

    return clean_answer(response.choices[0].message.content)


# =========================================================
# UI
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧬 BioASQ RAG Demo</div>
        <div class="hero-subtitle">
            Ask a biomedical question. Get an answer grounded in PubMed passages
            using dense retrieval and a large language model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pipeline">
        <span class="pipeline-step">User question</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">S-PubMedBert</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">FAISS top-5</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">LLaMA-3.3-70B</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-step">Grounded answer</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        <strong>⚠️ Informational use only.</strong>
        This system is provided for research and educational purposes.
        It is not a substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)


example_questions = [
    "What is the role of BRCA1 in breast cancer?",
    "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
    "Which syndrome is associated with mutations in the LYST gene?",
    "What organism causes Rhombencephalitis?",
    "Is the protein Papilin secreted?",
]


with st.form("rag_form"):
    question = st.text_area(
        "Enter your biomedical question:",
        placeholder="e.g., What is the role of BRCA1 in breast cancer?",
        height=100,
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        submitted = st.form_submit_button("Ask", type="primary")

    with col2:
        top_k = st.slider("Number of retrieved passages", 1, 10, DEFAULT_TOP_K)


st.markdown("#### Try an example")

selected_example = st.selectbox(
    "Example questions",
    [""] + example_questions,
    label_visibility="collapsed",
)

if selected_example and not question:
    question = selected_example
    submitted = True


if submitted:
    question = (question or "").strip()

    if not question:
        st.warning("Please enter a biomedical question.")
    else:
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                retrieved = retrieve(question, k=top_k)
                prompt = build_prompt(question, retrieved)
                answer = generate_answer(prompt)

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-label">Answer</div>
                        <div class="answer-body">{answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"📖 Retrieved evidence top-{top_k}", expanded=False):
                    for _, row in retrieved.iterrows():
                        rank = int(row["rank"])
                        pid = row[PASSAGE_ID_COL]
                        score = float(row["retrieval_score"])
                        text = str(row[PASSAGE_TEXT_COL]).strip()

                        st.markdown(
                            f"""
                            <div class="passage-card">
                                <div class="passage-header">
                                    <span class="passage-rank">#{rank}</span>
                                    <span>PMID <code>{pid}</code></span>
                                    <span>score <code>{score:.4f}</code></span>
                                </div>
                                <div class="passage-body">{text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except Exception as exc:
                st.error(f"Something went wrong: {type(exc).__name__}: {exc}")


st.markdown(
    """
    <div class="footer">
        <div>
            <strong>Retriever:</strong> <code>pritamdeka/S-PubMedBert-MS-MARCO</code> + FAISS
            &nbsp;·&nbsp;
            <strong>Generator:</strong> <code>llama-3.3-70b-versatile</code> via Groq
        </div>
        <div style="margin-top: 6px;">
            Built as a course project demonstrating Retrieval-Augmented Generation over the BioASQ corpus.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
