"""
streamlit_app.py
FinSight — Grab Holdings Financial Filings QA Assistant
Full Streamlit application with all three pipeline variants.

Run: streamlit run app/streamlit_app.py
"""

import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight — Grab Holdings",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background: #f0f7ff;
        border-left: 4px solid #2E86C1;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .citation-ref {
        font-weight: 700;
        color: #1a6396;
    }
    .citation-meta {
        color: #555;
        font-size: 0.82rem;
    }
    .citation-snippet {
        color: #333;
        font-style: italic;
        margin-top: 4px;
    }
    .chunk-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .score-badge {
        background: #e8f4fd;
        color: #1a6396;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .variant-badge-v1 { background: #fef9e7; color: #7d6608; }
    .variant-badge-v2 { background: #eafaf1; color: #1e8449; }
    .variant-badge-v3 { background: #f4ecf7; color: #7d3c98; }
    .disclaimer {
        font-size: 0.78rem;
        color: #888;
        border-top: 1px solid #eee;
        padding-top: 8px;
        margin-top: 12px;
    }
    .stButton > button {
        background-color: #1B4F72;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        background-color: #2E86C1;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models — this takes ~30 seconds on first run...")
def load_pipelines():
    """Load all three pipeline variants once and cache them across sessions."""
    from src.pipeline.baseline import BaselinePipeline
    from src.pipeline.advanced_a import AdvancedAPipeline
    from src.pipeline.advanced_b import AdvancedBPipeline

    return {
        "V1 — Baseline (Dense)": BaselinePipeline(),
        "V2 — Advanced A (Dense + Rerank)": AdvancedAPipeline(),
        "V3 — Advanced B (Hybrid + Rerank)": AdvancedBPipeline(),
    }


@st.cache_data(show_spinner=False)
def get_corpus_info():
    """Return basic corpus statistics for sidebar display."""
    from src.utils.config_loader import load_config
    cfg = load_config()
    docs = cfg.get("documents", [])
    processed_dir = Path(cfg["paths"]["processed_data"])

    total_chunks = 0
    doc_count = 0
    for doc in docs:
        jf = processed_dir / f"{doc['id']}.json"
        if jf.exists():
            import json as _json
            with open(jf) as f:
                chunks = _json.load(f)
            total_chunks += len(chunks)
            doc_count += 1

    return {
        "doc_count": doc_count,
        "total_docs_configured": len(docs),
        "total_chunks": total_chunks,
    }


# ── Helper renderers ──────────────────────────────────────────────────────────

def render_citation_card(cit: dict):
    """Render one citation as a styled HTML card."""
    ref = cit.get("ref", "")
    doc_type = cit.get("doc_type", "")
    period = cit.get("fiscal_period", "")
    date = cit.get("filing_date", "")
    page = cit.get("page_number", "")
    source = cit.get("source_file", "")
    section = cit.get("section_title", "")
    snippet = cit.get("snippet", "")
    score = cit.get("score", None)

    section_html = f" · <em>{section}</em>" if section else ""
    score_html = f'<span class="score-badge">score {score:.3f}</span>' if score else ""

    html = f"""
    <div class="citation-card">
        <span class="citation-ref">[{ref}]</span>{score_html}
        <div class="citation-meta">
            {doc_type} · {period} · Filed {date} · Page {page}{section_html}<br>
            <code style="font-size:0.78rem">{source}</code>
        </div>
        <div class="citation-snippet">"{snippet}"</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_chunk_card(chunk: dict, idx: int):
    """Render one retrieved chunk in the evidence panel."""
    meta = chunk.get("metadata", {})
    score = chunk.get("rerank_score", chunk.get("score", 0))
    retriever = chunk.get("retriever", "")
    found_by = chunk.get("found_by", "")
    text = chunk.get("text", "")

    source_label = f"{meta.get('doc_type','')} | {meta.get('fiscal_period','')} | p.{meta.get('page_number','')}"
    score_label = f"score: {score:.4f} | {retriever}"
    if found_by:
        score_label += f" | found_by: {found_by}"

    with st.expander(f"Chunk {idx+1}: {meta.get('chunk_id', 'unknown')} | {source_label}", expanded=False):
        st.caption(score_label)
        st.text(text)


def variant_description(method: str) -> str:
    descriptions = {
        "V1 — Baseline (Dense)": "Dense vector retrieval only. Fast, good for direct factual lookups.",
        "V2 — Advanced A (Dense + Rerank)": "Dense retrieval with cross-encoder reranking. Better precision, fewer hallucinations.",
        "V3 — Advanced B (Hybrid + Rerank)": "BM25 + dense retrieval fused by RRF, then reranked. Best for keyword-rich and complex queries.",
    }
    return descriptions.get(method, "")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    method = st.radio(
        "Retrieval Method",
        options=[
            "V1 — Baseline (Dense)",
            "V2 — Advanced A (Dense + Rerank)",
            "V3 — Advanced B (Hybrid + Rerank)",
        ],
        index=2,
        help="Select the RAG variant to use for this query",
    )
    st.caption(variant_description(method))

    st.divider()

    # Corpus info
    st.markdown("### 📚 Corpus")
    try:
        info = get_corpus_info()
        st.metric("Documents indexed", f"{info['doc_count']}/{info['total_docs_configured']}")
        st.metric("Total chunks", f"{info['total_chunks']:,}")
    except Exception as e:
        st.warning(f"Corpus info unavailable: {e}")

    st.divider()

    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What was Grab's total revenue for FY2023?",
        "How did adjusted EBITDA change from FY2022 to FY2023?",
        "What are the top regulatory risks disclosed in the FY2023 20-F?",
        "What guidance did management give for FY2024 GMV?",
        "How did the Deliveries GMV grow in Q3 2024 vs Q3 2023?",
    ]
    selected_example = st.selectbox("Load an example:", ["— select —"] + example_questions)

    st.divider()
    st.markdown(
        '<div class="disclaimer">⚠️ For research purposes only.<br>'
        'This tool does not constitute financial advice.<br>'
        'All answers sourced from official Grab Holdings filings only.</div>',
        unsafe_allow_html=True,
    )


# ── Main panel ────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📊 FinSight</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Grab Holdings Financial Filings Q&A — '
    'Answers sourced exclusively from official 20-F, 6-K, and investor presentation documents.</div>',
    unsafe_allow_html=True,
)

# Question input — pre-fill from example selector
default_q = selected_example if selected_example != "— select —" else ""
question = st.text_area(
    "Ask a question about Grab Holdings:",
    value=default_q,
    height=90,
    placeholder="e.g. What was Grab's total revenue in FY2023? What are the key risks in the 20-F?",
    key="question_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Submit", type="primary", use_container_width=True)
with col2:
    if question.strip():
        st.caption(f"Using: **{method}**")

# ── Handle submission ─────────────────────────────────────────────────────────

if submit and question.strip():
    try:
        pipelines = load_pipelines()
    except Exception as e:
        st.error(
            f"Failed to load pipelines: {e}\n\n"
            f"Make sure you have:\n"
            f"1. Installed all requirements: `pip install -r requirements.txt`\n"
            f"2. Added your OPENAI_API_KEY to `.env`\n"
            f"3. Run the ingestion and index build: `python scripts/ingest_all.py && python scripts/build_index.py`"
        )
        st.stop()

    pipeline = pipelines[method]

    with st.spinner(f"Retrieving evidence and generating answer via {method}..."):
        t0 = time.time()
        try:
            result = pipeline.ask(question)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
        total_latency = (time.time() - t0) * 1000

    # ── Evidence confidence check ─────────────────────────────────────────────
    from src.utils.config_loader import load_config
    cfg = load_config()
    threshold = cfg["retrieval"].get("weak_evidence_threshold", 0.3)

    chunks = result.get("retrieved_chunks", [])
    max_score = max(
        (c.get("rerank_score", c.get("score", 0.0)) for c in chunks),
        default=0.0,
    )

    if result.get("insufficient_evidence"):
        st.warning(
            "⚠️ **Insufficient evidence**: The system could not find strong supporting "
            "evidence in the Grab filings for this question."
        )
    elif max_score < threshold:
        st.warning(
            f"⚠️ **Low confidence** (top evidence score: {max_score:.3f} < {threshold}): "
            "Retrieved evidence may not directly address this question. "
            "Review the citations carefully."
        )

    if result.get("error"):
        st.error(f"Generation error: {result['error']}")

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Answer")

    # Render answer with styled citation markers
    try:
        from src.generation.citation_formatter import annotate_answer_html
        answer_html = annotate_answer_html(
            result.get("answer", ""),
            result.get("citations", [])
        )
        st.markdown(answer_html, unsafe_allow_html=True)
    except Exception:
        st.markdown(result.get("answer", ""))

    # Metadata row
    col_lat, col_model, col_tok, col_var = st.columns(4)
    with col_lat:
        st.caption(f"⏱ {total_latency:.0f}ms total")
    with col_model:
        st.caption(f"🤖 {result.get('model', 'N/A')}")
    with col_tok:
        st.caption(f"🪙 {result.get('total_tokens', 0)} tokens")
    with col_var:
        st.caption(f"🔬 {result.get('variant', '')}")

    # ── Citations ──────────────────────────────────────────────────────────────
    citations = result.get("citations", [])
    if citations:
        st.markdown("---")
        with st.expander(f"📎 Citations ({len(citations)})", expanded=True):
            for cit in citations:
                render_citation_card(cit)
    else:
        st.info("No citations extracted — the answer may not reference specific document passages.")

    # ── Retrieved Evidence ────────────────────────────────────────────────────
    if chunks:
        st.markdown("---")
        with st.expander(f"🔍 Retrieved Evidence ({len(chunks)} chunks)", expanded=False):
            st.caption(
                f"Top evidence score: **{max_score:.4f}** | "
                f"Retriever: **{result.get('variant', '')}**"
            )
            for i, chunk in enumerate(chunks):
                render_chunk_card(chunk, i)

    # ── Debug panel (hidden by default) ──────────────────────────────────────
    with st.expander("🛠 Debug Info", expanded=False):
        debug_data = {
            "variant": result.get("variant"),
            "model": result.get("model"),
            "latency_ms": result.get("latency_ms"),
            "generation_latency_ms": result.get("generation_latency_ms"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "n_chunks_retrieved": len(chunks),
            "max_evidence_score": round(max_score, 4),
            "insufficient_evidence": result.get("insufficient_evidence"),
            "n_citations": len(citations),
            "fusion_stats": result.get("fusion_stats"),
        }
        st.json(debug_data)

elif submit and not question.strip():
    st.warning("Please enter a question before submitting.")

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.info(
        "👆 Enter a question above about Grab Holdings' financials, strategy, or risk factors.\n\n"
        "This system answers **only** from official Grab filings — no external data is used."
    )

    # Show sample questions as clickable hints
    st.markdown("**Try these questions:**")
    cols = st.columns(2)
    sample_qs = [
        ("📈 Revenue", "What was Grab's total revenue for FY2023?"),
        ("💹 EBITDA", "How did Grab's adjusted EBITDA evolve from FY2022 to FY2023?"),
        ("⚠️ Risks", "What are the top three regulatory risks in the FY2023 20-F?"),
        ("🗺 Strategy", "What are Grab's strategic priorities as stated at Investor Day 2023?"),
        ("📊 Segments", "What was the GMV breakdown by segment in FY2023?"),
        ("💰 Cash", "What was Grab's cash position at end of FY2023?"),
    ]
    for i, (label, q) in enumerate(sample_qs):
        col = cols[i % 2]
        with col:
            if st.button(f"{label}: {q[:45]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state["question_input"] = q
                st.rerun()
