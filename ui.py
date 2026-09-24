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
        f"<div class='badge-card'><b>LLM Engine:</b> LLaMA-3.3-70B (Groq)<br>"
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

# Suggested Query Chips
st.markdown("**Quick Topics:**")
col1, col2, col3 = st.columns(3)
quick_query = None
if col1.button("⚡ Direct Preference Optimization in LLMs"):
    quick_query = "Direct Preference Optimization vs RLHF for Large Language Model Alignment"
if col2.button("🔬 Mechanistic Interpretability in Transformers"):
    quick_query = "Mechanistic Interpretability of Attention Heads and Induction Heads in Transformers"
if col3.button("🐍 State Space Models (Mamba) vs Attention"):
    quick_query = "State Space Models Mamba versus Linear Attention Transformers for Long Sequences"

query_input = st.text_input(
    "Enter Research Topic or Scientific Hypothesis:",
    value=quick_query if quick_query else "",
    placeholder="e.g., Parameter Efficient Fine Tuning LoRA vs QLoRA tradeoffs"
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

        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown("##### ⏳ Execution Progress")
            status_box = st.status("Executing Multi-Agent Pipeline...", expanded=True)
            with status_box:
                st.write("🔍 **Phase 1:** Scouting arXiv & Semantic Scholar for peer-reviewed papers...")
                time.sleep(1)
                st.write("📥 **Phase 2:** Downloading open-access PDFs & indexing with layout-aware PyMuPDF...")
                time.sleep(1)
                st.write("⚖️ **Phase 3:** Extracting methodologies, datasets & running Reciprocal Rank Fusion...")
                time.sleep(1)
                st.write("💡 **Phase 4:** Synthesizing research gaps, scaling limits & future directions...")

        start_time = time.time()
        try:
            with st.spinner("Agents are actively analyzing literature... (typically takes 30-60s)"):
                report_content = run_research(query=query_input.strip(), save_report=True)

            elapsed = time.time() - start_time
            status_box.update(label=f"✅ Research Complete in {elapsed:.1f}s!", state="complete", expanded=False)

            st.success(f"Investigation completed successfully in {elapsed:.1f} seconds!")

            # Output Presentation Tabs
            tab1, tab2 = st.tabs(["📄 Executive Research Dossier", "💾 Raw Markdown"])

            with tab1:
                st.markdown(report_content)

            with tab2:
                st.code(report_content, language="markdown")

            st.download_button(
                label="📥 Export Report as Markdown (.md)",
                data=report_content,
                file_name=f"research_{int(time.time())}.md",
                mime="text/markdown"
            )

        except Exception as e:
            status_box.update(label="❌ Pipeline Failed", state="error")
            st.error(f"Error during research execution: {e}")
