import os
import re
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRAG:
    """
    Production-grade Hybrid RAG Engine combining:
    1. Layout-aware PDF chunking with page-level attribution
    2. Dense Vector Retrieval via Sentence-Transformers & ChromaDB
    3. Sparse Lexical Retrieval via BM25Okapi
    4. Reciprocal Rank Fusion (RRF) for balanced multi-stage retrieval
    """

    def __init__(
        self,
        collection_name: str = "academic_research",
        persist_directory: str = "chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        print(f"[HybridRAG] Initializing Embedder: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)

        print(f"[HybridRAG] Connecting to ChromaDB at: {persist_directory}...")
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Academic papers hybrid index"}
        )

        # In-memory BM25 index components
        self.bm25_documents: List[str] = []
        self.bm25_metadatas: List[Dict[str, Any]] = []
        self.bm25_ids: List[str] = []
        self.bm25_index: Optional[BM25Okapi] = None

        self._rebuild_bm25_from_chroma()

    def _rebuild_bm25_from_chroma(self):
        """Loads all existing items from ChromaDB into BM25 memory index."""
        try:
            stored = self.collection.get()
            if stored and stored.get("documents"):
                self.bm25_documents = stored["documents"]
                self.bm25_metadatas = stored["metadatas"] or [{}] * len(self.bm25_documents)
                self.bm25_ids = stored["ids"]
                tokenized_corpus = [doc.lower().split() for doc in self.bm25_documents]
                if tokenized_corpus:
                    self.bm25_index = BM25Okapi(tokenized_corpus)
        except Exception as e:
            print(f"[HybridRAG] Warning rebuilding BM25 index: {e}")

    def chunk_pdf(
        self,
        pdf_path: str,
        chunk_size_words: int = 400,
        overlap_words: int = 80
    ) -> List[Dict[str, Any]]:
        """
        Parses a PDF using PyMuPDF page-by-page, chunking while tracking
        exact page numbers and document titles for verifiable citations.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        base_name = os.path.basename(pdf_path)
        clean_title = os.path.splitext(base_name)[0]

        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Clean hyphens at line wraps, multiple spaces, and noise
            text = re.sub(r"-\n\s*", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            if not text or len(text) < 50:
                continue

            words = text.split()
            step = max(1, chunk_size_words - overlap_words)

            for i in range(0, len(words), step):
                chunk_words = words[i : i + chunk_size_words]
                chunk_text = " ".join(chunk_words)

                if len(chunk_text.strip()) > 80:
                    chunk_id = f"{clean_title}_p{page_num + 1}_{i}"
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source_file": base_name,
                            "title": clean_title,
                            "page": page_num + 1,
                            "word_count": len(chunk_words)
                        }
                    })

        doc.close()
        return chunks

    def ingest_pdf(self, pdf_path: str, title: Optional[str] = None) -> int:
        """
        Extracts, chunks, embeds, and indexes a PDF into both ChromaDB and BM25.
        """
        chunks = self.chunk_pdf(pdf_path)
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        if title:
            for m in metadatas:
                m["paper_title"] = title

        # Dense Vector embeddings
        embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()

        # Add to Chroma
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        # Update BM25
        for cid, doc, meta in zip(ids, documents, metadatas):
            if cid not in self.bm25_ids:
                self.bm25_ids.append(cid)
                self.bm25_documents.append(doc)
                self.bm25_metadatas.append(meta)

        tokenized_corpus = [doc.lower().split() for doc in self.bm25_documents]
        if tokenized_corpus:
            self.bm25_index = BM25Okapi(tokenized_corpus)

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Executes Reciprocal Rank Fusion (RRF) between Dense (Semantic) and Sparse (BM25) searches.
        """
        total_docs = len(self.bm25_documents)
        if total_docs == 0:
            return []

        # 1. Dense Search
        query_embedding = self.embedder.encode([query]).tolist()
        fetch_k = min(top_k * 3, total_docs)
        dense_res = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k
        )

        dense_ranked_ids = []
        if dense_res and dense_res.get("ids") and dense_res["ids"][0]:
            dense_ranked_ids = dense_res["ids"][0]

        # 2. Sparse BM25 Search
        sparse_ranked_ids = []
        if self.bm25_index:
            tokens = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokens)
            sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            sparse_ranked_ids = [self.bm25_ids[i] for i in sorted_indices[:fetch_k] if bm25_scores[i] > 0]

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(dense_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_weight * (1.0 / (rrf_k + rank + 1))

        for rank, doc_id in enumerate(sparse_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + sparse_weight * (1.0 / (rrf_k + rank + 1))

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        id_to_meta = {cid: meta for cid, meta in zip(self.bm25_ids, self.bm25_metadatas)}
        id_to_doc = {cid: doc for cid, doc in zip(self.bm25_ids, self.bm25_documents)}

        final_chunks = []
        for doc_id, score in sorted_results:
            final_chunks.append({
                "id": doc_id,
                "text": id_to_doc.get(doc_id, ""),
                "metadata": id_to_meta.get(doc_id, {}),
                "rrf_score": score
            })

        return final_chunks

    def format_citation_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks with citations suitable for LLM grounding."""
        if not search_results:
            return "No relevant literature context found in vector knowledge base."

        formatted = []
        for i, item in enumerate(search_results, 1):
            meta = item.get("metadata", {})
            source = meta.get("paper_title") or meta.get("title") or meta.get("source_file", "Unknown")
            page = meta.get("page", "?")
            formatted.append(
                f"[Document {i}] - Source: {source} (Page {page})\n"
                f"Context Excerpt:\n{item['text']}\n"
            )

        return "\n------------------------------------\n".join(formatted)
