from langchain.chains import RetrievalQA
from scripts.llm_setup import get_llm
from scripts.chromadb_setup import get_chroma_db, search_chroma
import logging
from typing import Dict, Any, List
from pathlib import Path

# Try to reuse your chromadb helper if available
try:
    from scripts.chromadb_setup import search_chroma
except Exception:
    search_chroma = None

logging.basicConfig(level=logging.INFO, format="%(message)s")

def build_rag_chain():
    """
    Builds a Retrieval-Augmented Generation (RAG) chain using Groq LLM and ChromaDB retriever.

    Returns:
        RetrievalQA: A LangChain RetrievalQA chain instance.
    """
    llm = get_llm()
    db = get_chroma_db()
    retriever = db.as_retriever()
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Optional: helpful for debugging or UI
    )
    return rag_chain

def _normalize_docs(docs) -> List[Dict[str, Any]]:
    normalized = []
    for d in docs:
        # LangChain Document objects
        try:
            content = getattr(d, "page_content", None) or getattr(d, "content", None) or getattr(d, "text", None)
            metadata = getattr(d, "metadata", None) or getattr(d, "meta", None) or {}
            normalized.append({"text": content, "metadata": metadata})
        except Exception:
            # fallback dict-like
            if isinstance(d, dict):
                normalized.append({"text": d.get("text"), "metadata": d.get("metadata", {})})
            else:
                normalized.append({"text": str(d), "metadata": {}})
    return normalized

def run_query(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Very small RAG runner: uses search_chroma() if available, otherwise
    looks for a local JSON store in data/chromadb_store.json (fallback).
    Returns {'result': <str>, 'source_documents': [docs...]}
    """
    logging.info(f"Running RAG query: {query}")
    docs = search_chroma(query, k=k) or []
    hits = _normalize_docs(docs)
    context = "\n\n".join(h["text"] for h in hits if h["text"])
    prompt = f"""Use the following retrieved context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer concisely:"""
    llm = get_llm()
    try:
        answer = llm(prompt)
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        answer = f"[LLM_ERROR] {e}"
    return {"result": answer, "source_documents": hits}