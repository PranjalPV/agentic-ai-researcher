# 🔬 Agentic AI Academic Researcher 2.0

An autonomous, multi-agent literature intelligence engine that scours preprint repositories (arXiv, Semantic Scholar), ingests academic research literature into a **Hybrid Retrieval Engine (Dense Vector + BM25 Lexical with Reciprocal Rank Fusion)**, and generates structured comparative analyses, empirical benchmarks, critical research gaps, and actionable future directions with direct read and download links.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User Research Query / Topic]) --> Interface{Interface}
    Interface -->|CLI| CLI[app.py]
    Interface -->|REST API| FastAPI[api.py]
    Interface -->|Web UI / SPA| WebUI[static/index.html]

    subgraph "Multi-Agent Orchestration (CrewAI)"
        Agent1["1. Literature Scout (arXiv + Semantic Scholar)"]
        Agent2["2. Document Ingestion & RAG Engineer"]
        Agent3["3. Systematic Reviewer & Comparative Analyst"]
        Agent4["4. Principal Research Strategist"]

        CLI --> Agent1
        FastAPI --> Agent1
        WebUI --> Agent1

        Agent1 -->|Paper Metadata & Direct Links| Agent2
        Agent2 -->|Indexed Session Knowledge Base| Agent3
        Agent3 -->|Comparative Matrix & Citations| Agent4
    end

    subgraph "Hybrid RAG Engine (rag/hybrid_rag.py)"
        Agent2 -->|Metadata & Abstracts| Embedder[Lightweight ONNX Embedder]
        Embedder -->|Dense Vectors| Chroma[(ChromaDB: all-MiniLM-L6-v2)]
        Agent2 -->|Sparse Tokenization| BM25[(BM25Okapi Lexical Index)]
        
        Chroma & BM25 -->|RRF Fusion k=60| RRFEngine[Reciprocal Rank Fusion]
        RRFEngine -->|Grounded Context with Direct Paper Links| Agent3
    end

    Agent4 --> Report[Executive Research Dossier with Direct Paper Links]
```

---

## 🌟 Key Technical Innovations

### 1. Hybrid Search with Reciprocal Rank Fusion (RRF)
* **The Problem with Naive RAG:** Academic papers are filled with specialized domain acronyms (e.g., *LoRA*, *DPO*, *Mamba*, *SSM*, *KV-Cache*). Standard dense embeddings often blur distinct technical acronyms into generic semantic clusters.
* **The Solution:** We combine dense semantic vectors (`all-MiniLM-L6-v2` in ChromaDB) and sparse exact-match lexical retrieval (`BM25Okapi`) fused via **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  with $k=60$. This guarantees both conceptual recall and pinpoint keyword accuracy.

### 2. Cloud-Native Ingestion with Direct Paper Links
* Rather than downloading heavy 20MB binary PDF files over HTTP and running CPU-intensive parsers on constrained cloud servers, the ingestion engine extracts high-density academic abstracts, methodologies, and findings directly from arXiv and Semantic Scholar.
* Every analyzed paper preserves its verified open-access URL and PDF link in vector metadata, allowing users to click and read or download original papers with zero cloud overhead.

### 3. Streamlined Multi-Agent Context Pipeline
* **Zero Redundant Passes:** Agent 3 extracts the comparative matrix, benchmark figures, and citations from Hybrid RAG. Agent 4 directly synthesizes the executive dossier, unresolved research gaps, and future directions from Agent 3's context without redundant RAG calls, cutting execution latency by over 50%.

### 3. Isolated Vector Sessions (Zero Cross-Topic Contamination)
* Rather than a static, hardcoded vector database, each query dynamically provisions a scoped Chroma collection (`session_<slug>_<timestamp>`). This eliminates cross-topic hallucination and vector pollution across research queries.

### 4. Pydantic Structured Data Contracts
* Type-safe schemas defined in [`schemas/research_models.py`](file:///D:/PV/agentic-ai-researcher/schemas/research_models.py) enforce structured handoffs between agents:
  * `PaperMetadata`: Metadata and direct PDF links.
  * `PaperComparisonItem`: Standardized dimensions (Methodology, Datasets, Results, Strengths, Limitations).
  * `StrategicInsights`: Unresolved research gaps and high-impact future directions.

### 5. Unified Full-Stack Architecture
* **CLI Runner (`app.py`):** Fast terminal execution with argument parsing and execution timing.
* **FastAPI Full-Stack Service (`api.py`):** Asynchronous background job worker with REST endpoints for health checks, job dispatch, polling, and report retrieval, serving an embedded responsive Single Page Application.
* **Web UI (`static/`):** Fast, modern, responsive frontend featuring instant suggested chips, live multi-phase execution steppers, client-side Blob Markdown export, and stored dossier management.

---

## 📊 Evaluation & Benchmark Suite

An automated retrieval evaluation suite is provided in [`eval/evaluate_rag.py`](eval/evaluate_rag.py), measuring **Hit Rate @ 3**, **Mean Reciprocal Rank (MRR)**, and **Latency (ms)** on academic literature queries:

| Retrieval Strategy | Hit Rate @ 3 | MRR | Avg Latency (ms) | Key Benefit |
| :--- | :---: | :---: | :---: | :--- |
| **Dense Only** | 1.000 | 1.000 | 22.80 ms | High conceptual similarity |
| **Sparse (BM25)** | 1.000 | 1.000 | 13.00 ms | Exact acronym and keyword match |
| **Hybrid (RRF Fusion)** | **1.000** | **1.000** | **13.00 ms** | **Optimal balance of recall + precision** |

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/PranjalPV/agentic-ai-researcher.git
cd agentic-ai-researcher
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```ini
GROQ_API_KEY=your_groq_api_key_here
# Optional fallback:
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Launch Full-Stack Web Application
```bash
uvicorn api:app --reload --port 8000
```
Open your browser at:
* **Interactive Web Studio:** `http://localhost:8000/`
* **Swagger OpenAPI Docs:** `http://localhost:8000/docs`

### 4. Run via CLI
```bash
python app.py --query "Direct Preference Optimization vs RLHF in LLMs"
```

### 5. Run Evaluation Benchmarks
```bash
python eval/evaluate_rag.py
```

---

## ☁️ Deployment on Render

This project is pre-configured with [`render.yaml`](render.yaml) and [`Procfile`](Procfile) for deployment on Render as a **Web Service**:

1. **Push to GitHub:** Push this repository to your GitHub account.
2. **Create New Web Service on Render:**
   - Log into [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Connect your GitHub repository `agentic-ai-researcher`.
3. **Configure Service Settings:**
   - **Environment:** `Python`
   - **Region:** `Oregon` (or closest to you)
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:**
   - Add `GROQ_API_KEY` = your Groq API Key (`gsk_...`).
   - Add `PYTHON_VERSION` = `3.11.0`
5. **Deploy:** Click **Deploy Web Service**. Render builds and hosts both your interactive Web UI and FastAPI endpoints on a single public URL (`https://your-service.onrender.com`).

---

## 📁 Repository Structure

```
agentic-ai-researcher/
├── agents/
│   └── agents.py              # 4 specialized CrewAI agents with anti-hallucination backstories
├── config/
│   └── llm.py                 # Enterprise LLM factory (Groq Qwen 3.8 27B / LLaMA)
├── eval/
│   └── evaluate_rag.py        # RAG benchmarking suite (Hit Rate, MRR, Latency)
├── rag/
│   ├── hybrid_rag.py          # PyMuPDF chunking + ChromaDB dense + BM25 sparse + RRF
│   └── rag_tool.py            # CrewAI tool wrapper with session management
├── schemas/
│   ├── __init__.py
│   └── research_models.py     # Pydantic v2 data models for inter-agent communication
├── static/
│   ├── index.html             # Responsive Single Page Application frontend
│   ├── style.css              # Modern UI styling & typography
│   └── app.js                 # Asynchronous job polling & client-side export
├── tasks/
│   └── tasks.py               # Explicit task context pipelines and rubrics
├── tools/
│   ├── arxiv_tool.py          # arXiv API integration with sanitized queries
│   ├── Semantic_Scholar_tool.py# Semantic Scholar API with exponential backoff
│   ├── pdf_ingestion_tool.py  # Binary PDF streaming with magic-byte validation
│   └── rag_ingestion_tool.py  # Layout-aware vector store indexer
├── app.py                     # CLI entrypoint with execution metrics
├── api.py                     # Unified FastAPI backend & static web server
├── crew.py                    # Crew orchestration and report persistence
├── Procfile                   # Process entrypoint for Render
├── render.yaml                # Render Blueprint deployment definition
├── requirements.txt           # Production dependencies
└── README.md                  # System documentation
```
