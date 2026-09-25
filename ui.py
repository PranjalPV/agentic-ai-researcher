import os
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Set page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Agentic AI Academic Researcher",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Synchronize Streamlit Secrets to os.environ for Cloud Deployments
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)


def load_past_reports():
    if not os.path.exists(REPORTS_DIR):
        return []
    files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".md")]
    return sorted(files, reverse=True)


# Sidebar Configuration & Telemetry
with st.sidebar:
    st.markdown("### ⚙️ System Telemetry")
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    # Allow direct API key entry if not already set via environment or secrets
    if not groq_api_key:
        user_key = st.text_input("Enter Groq API Key:", type="password", help="Get a free key at console.groq.com")
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key
            groq_api_key = user_key

    groq_ready = bool(groq_api_key)
    st.markdown(
        f"<div class='badge-card'><b>LLM Engine:</b> Qwen 3.8 27B (Groq)<br>"
        f"<b>Status:</b> {'🟢 Online' if groq_ready else '🔴 Missing Key'}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='badge-card'><b>Hybrid Retrieval:</b><br>"
        "• Dense: all-MiniLM-L6-v2 (ChromaDB)<br>"
        "• Sparse: BM25Okapi Lexical<br>"
        "• Fusion: Reciprocal Rank Fusion (k=60)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='badge-card'><b>Multi-Agent Crew:</b><br>"
        "1. Literature Scout<br>"
        "2. Ingestion & RAG Engineer<br>"
        "3. Comparative Analyst<br>"
        "4. Research Strategist</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 📂 Stored Dossiers")
    reports = load_past_reports()
    if reports:
        selected_report = st.selectbox("Select Past Report:", reports)
        if selected_report:
            with open(os.path.join(REPORTS_DIR, selected_report), "r", encoding="utf-8") as f:
                report_text = f.read()
            st.download_button(
                label="📥 Download Selected Report",
                data=report_text,
                file_name=selected_report,
                mime="text/markdown"
            )
    else:
        st.caption("No reports generated yet.")

# Main Application Interface
st.markdown("<div class='main-header'>Agentic AI Academic Researcher 2.0</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Autonomous Multi-Agent Literature Discovery, Layout-Aware Hybrid RAG, & Systematic Comparative Synthesis</div>",
    unsafe_allow_html=True
)

# Initialize session state for persistent research output across reruns and downloads
if "current_report" not in st.session_state:
    st.session_state["current_report"] = None
if "current_topic" not in st.session_state:
    st.session_state["current_topic"] = None
if "current_elapsed" not in st.session_state:
    st.session_state["current_elapsed"] = None
if "current_timestamp" not in st.session_state:
    st.session_state["current_timestamp"] = None

# Topic input with example prompt guidance
query_input = st.text_input(
    "Enter Research Topic or Scientific Hypothesis:",
    placeholder="e.g., Direct Preference Optimization vs RLHF for Large Language Models",
    help="Enter any topic in AI, computer science, biology, or scientific literature."
)
st.caption(
    "💡 **Example topics:** "
    "`Direct Preference Optimization in LLMs` • "
    "`Mechanistic Interpretability in Transformers` • "
    "`State Space Models (Mamba) vs Attention` • "
    "`LoRA vs QLoRA Tradeoffs`"
)

launch_btn = st.button("🚀 Launch Autonomous Research Crew", type="primary", use_container_width=True)

if launch_btn:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please provide a valid GROQ_API_KEY in the sidebar or via Streamlit Secrets before launching.")
    elif not query_input.strip():
        st.warning("Please enter a research topic before launching the crew.")
    else:
        # Lazy import of crew pipeline to ensure environment variables are populated
        from crew import run_research

        st.info(f"Initiating multi-agent literature review for: **{query_input.strip()}**")

        status_box = st.status("Executing Multi-Agent Pipeline...", expanded=True)
        with status_box:
            st.write("🔍 **Phase 1:** Scouting arXiv & Semantic Scholar for peer-reviewed papers...")
            st.write("📥 **Phase 2:** Downloading open-access PDFs & indexing with layout-aware PyMuPDF...")
            st.write("⚖️ **Phase 3:** Extracting methodologies, datasets & running Reciprocal Rank Fusion...")
            st.write("💡 **Phase 4:** Synthesizing research gaps, scaling limits & future directions...")

        start_time = time.time()
        try:
            with st.spinner("Agents are actively analyzing literature... (typically takes 30-60s)"):
                report_content = run_research(query=query_input.strip(), save_report=True)

            elapsed = time.time() - start_time
            status_box.update(label=f"✅ Research Complete in {elapsed:.1f}s!", state="complete", expanded=False)

            # Persist in session state so it survives download button reruns
            st.session_state["current_report"] = report_content
            st.session_state["current_topic"] = query_input.strip()
            st.session_state["current_elapsed"] = elapsed
            st.session_state["current_timestamp"] = int(time.time())

        except Exception as e:
            status_box.update(label="❌ Pipeline Failed", state="error")
            st.error(f"Error during research execution: {e}")

# Output Presentation Tabs (Persisted across downloads and interactions)
if st.session_state.get("current_report"):
    report_content = st.session_state["current_report"]
    report_topic = st.session_state.get("current_topic", "Research Topic")
    elapsed = st.session_state.get("current_elapsed", 0.0)
    ts = st.session_state.get("current_timestamp", int(time.time()))

    st.markdown("---")
    st.success(f"Investigation completed successfully for **{report_topic}** ({elapsed:.1f} seconds)!")

    tab1, tab2 = st.tabs(["📄 Executive Research Dossier", "💾 Raw Markdown"])

    with tab1:
        st.markdown(report_content)

    with tab2:
        st.code(report_content, language="markdown")

    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in report_topic[:25]).strip("_")
    st.download_button(
        label="📥 Export Report as Markdown (.md)",
        data=report_content,
        file_name=f"research_{safe_name}_{ts}.md",
        mime="text/markdown",
        key="download_current_report_btn"
    )

