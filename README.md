# 🤖 Agentic AI Research Assistant
 
An autonomous multi-agent AI pipeline that searches, downloads, analyzes, and generates insights from academic research papers — powered by CrewAI and LLaMA 3.3 70B via Groq.
 
---
 
## 🚀 Features
 
- **Autonomous Research** — Searches arXiv and Semantic Scholar for relevant papers on any topic
- **Auto PDF Ingestion** — Downloads papers and indexes them into a local vector database
- **Structured Comparison** — Compares papers across methodology, datasets, metrics, results, and limitations
- **Insight Generation** — Identifies research gaps, open challenges, and future directions
- **Multi-Agent Pipeline** — 4 specialized agents working sequentially via CrewAI
- **Local RAG** — Runs fully local embeddings and vector search using ChromaDB + Sentence Transformers
---
 
## 🛠️ Tech Stack
 
| Layer | Technology |
|---|---|
| Multi-Agent Framework | CrewAI |
| LLM (Agents) | Groq API — LLaMA 3.3 70B Versatile |
| LLM (RAG Queries) | Groq API — LLaMA 3.1 8B Instant |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (local) |
| Research APIs | arXiv API, Semantic Scholar API |
| PDF Handling | requests |
| Environment | python-dotenv |
 
---
 
## 📁 Project Structure
 
```
agentic-ai-researcher/
│
├── agents/
│   └── agents.py              # 4 CrewAI agents: Research, Ingestion, Comparison, Insight
│
├── tasks/
│   └── tasks.py               # Tasks assigned to each agent
│
├── tools/
│   ├── arxiv_tool.py          # Searches arXiv for paper PDFs
│   ├── Semantic_Scholar_tool.py  # Searches Semantic Scholar for open-access PDFs
│   ├── pdf_ingestion_tool.py  # Downloads PDFs locally
│   └── rag_ingestion_tool.py  # Ingests PDFs into ChromaDB vector store
│
├── rag/
│   └── rag_tool.py            # CrewAI RagTool configured with Groq + ChromaDB
│
├── config/
│   └── llm.py                 # LLM config — Groq LLaMA 3.3 70B
│
├── db_healthcare_research/    # ChromaDB vector store (auto-created, not committed)
├── temp_papers/               # Temporary downloaded PDFs (auto-created, not committed)
│
├── crew.py                    # Assembles all agents and tasks into a Crew
├── app.py                     # Entry point — takes user topic and kicks off the crew
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
└── README.md
```
 
---
 
## 🧠 How It Works
 
```
User enters a research topic
        ↓
🔍 Research Agent
   → Searches arXiv + Semantic Scholar
   → Returns list of PDF URLs
        ↓
📥 Ingestion Agent
   → Downloads PDFs to temp_papers/
   → Ingests into ChromaDB vector store
        ↓
📊 Comparison Agent
   → Queries RAG for paper content
   → Compares methodology, datasets, metrics, results, limitations
        ↓
💡 Insight Agent
   → Queries RAG for deeper analysis
   → Generates research gaps, open challenges, future directions
        ↓
Final Report printed to console
```
 
---
 
## 🤖 Agents
 
| Agent | Role | Tools |
|---|---|---|
| **Research Agent** | Finds relevant papers | arXiv Tool, Semantic Scholar Tool |
| **Ingestion Agent** | Downloads & indexes PDFs | PDF Ingestion Tool, RAG Ingestion Tool |
| **Comparison Agent** | Compares papers structurally | RAG Tool |
| **Insight Agent** | Generates research insights | RAG Tool |
 
---
 
## ⚙️ Setup & Installation
 
### Prerequisites
- Python 3.9+
- A Groq API key (free at [console.groq.com](https://console.groq.com))
---
 
### 1. Clone the repository
 
```bash
git clone https://github.com/YOUR-USERNAME/agentic-ai-researcher.git
cd agentic-ai-researcher
```
 
### 2. Create and activate a virtual environment
 
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# Mac/Linux
source venv/bin/activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Create a `.env` file
 
```
GROQ_API_KEY=your_groq_api_key_here
```
 
### 5. Run the app
 
```bash
python app.py
```
 
Enter any research topic when prompted:
```
Enter research topic: transformer models in medical imaging
```
 
---
 
## 🔒 Environment Variables
 
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLaMA models |
 
> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.
 
---
 
## 📦 Requirements
 
Key dependencies (add to `requirements.txt`):
 
```
crewai
crewai-tools
groq
sentence-transformers
chromadb
arxiv
requests
python-dotenv
```
 
---
 
## 🤝 Contributing
 
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
 
---
 
## 📄 License
 
This project is licensed under the MIT License.
 
