import os
import json
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Disable Chroma telemetry
os.environ.setdefault("CHROMA_TELEMETRY", "0")

CHROMA_DB_DIR = "chroma_db"
FALLBACK_STORE = Path(__file__).resolve().parent.parent / "data" / "chromadb_store.json"
FALLBACK_STORE.parent.mkdir(parents=True, exist_ok=True)

# Prefer new langchain_chroma package
Chroma = None
try:
    from langchain_chroma import Chroma
    logging.debug("Imported Chroma from langchain_chroma")
except Exception:
    try:
        from langchain_community.vectorstores import Chroma
        logging.debug("Imported Chroma from langchain_community.vectorstores")
    except Exception:
        try:
            from langchain.vectorstores import Chroma
        except Exception:
            Chroma = None

# Try to find Groq embeddings (unchanged)
_GroqEmbeddings = None
for mod_path in (
    "langchain_groq.embeddings",
    "langchain_groq",
    "langchain_community.embeddings.groq",
    "langchain_community.embeddings",
    "langchain.embeddings",
):
    try:
        mod = __import__(mod_path, fromlist=["GroqEmbeddings", "GroqEmbedding"])
        _GroqEmbeddings = getattr(mod, "GroqEmbeddings", None) or getattr(mod, "GroqEmbedding", None)
        if _GroqEmbeddings:
            logging.debug(f"Found Groq embeddings in {mod_path}")
            break
    except Exception:
        continue

# Prefer langchain_community HuggingFaceEmbeddings first (future-proof)
_HuggingFaceEmbeddings = None
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
    except Exception:
        _HuggingFaceEmbeddings = None

# direct sentence-transformers fallback
_SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:
    _SentenceTransformer = None

def _fake_embed(texts):
    def vec(s):
        s = (s or "")[:256]
        return [float(ord(c) % 97) / 97.0 for c in s.ljust(32, "\0")[:32]]
    return [vec(t) for t in texts]

def get_embedding_function():
    """
    Returns callable list[str] -> list[list[float]].
    Priority:
      1) GroqEmbeddings (if available + GROQ_API_KEY)
      2) HuggingFaceEmbeddings (langchain_community preferred) using EMBEDDING_MODEL env var
      3) sentence-transformers direct
      4) Fake deterministic embeddings
    """
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    groq_key = os.getenv("GROQ_API_KEY")
    if _GroqEmbeddings and groq_key:
        try:
            try:
                emb = _GroqEmbeddings(api_key=groq_key)
            except TypeError:
                emb = _GroqEmbeddings()
            if hasattr(emb, "embed_documents"):
                return lambda texts: emb.embed_documents(texts)
            if hasattr(emb, "embed"):
                return lambda texts: [emb.embed(t) for t in texts]
        except Exception as e:
            logging.warning(f"Failed to init Groq embeddings: {e}")

    if _HuggingFaceEmbeddings:
        try:
            logging.info(f"Using HuggingFaceEmbeddings ({model_name}).")
            hfe = _HuggingFaceEmbeddings(model_name=model_name)
            return lambda texts: hfe.embed_documents(texts)
        except Exception as e:
            logging.debug(f"HuggingFaceEmbeddings init failed: {e}")

    if _SentenceTransformer:
        try:
            logging.info(f"Using sentence-transformers model '{model_name}'.")
            st = _SentenceTransformer(model_name)
            return lambda texts: st.encode(texts).tolist() if hasattr(st.encode(texts), "tolist") else st.encode(texts)
        except Exception as e:
            logging.debug(f"SentenceTransformer init failed: {e}")

    logging.info("No real embedding provider available — using fake embeddings.")
    return _fake_embed

# Adapter so Chroma always gets an object with embed_documents/embed_query
class _EmbeddingAdapter:
    def __init__(self, fn):
        self._fn = fn
    def embed_documents(self, texts):
        return self._fn(texts)
    def embed_query(self, text):
        return self._fn([text])[0]
    def embed(self, texts):
        return self._fn(texts)

def _reinit_chroma_dir(persist_directory: str):
    """Remove chroma persist dir and related artifacts to allow reinitialization."""
    try:
        p = Path(persist_directory)
        if p.exists():
            logging.warning(f"Reinitializing Chroma persist directory: {p} (will be removed)")
            shutil.rmtree(p)
        # recreate empty dir
        p.mkdir(parents=True, exist_ok=True)
        logging.info("Chroma persist directory reinitialized.")
    except Exception as e:
        logging.error(f"Failed to reinitialize Chroma persist directory: {e}")

def get_chroma_db(persist_directory=CHROMA_DB_DIR):
    embedding_fn = get_embedding_function()
    embedding_obj = embedding_fn
    if callable(embedding_fn) and not hasattr(embedding_fn, "embed_documents"):
        embedding_obj = _EmbeddingAdapter(embedding_fn)

    if Chroma is None:
        logging.warning("Chroma import not available; using fallback JSON store.")
        return None, embedding_fn

    try:
        db = Chroma(
            collection_name="rag_docs",
            embedding_function=embedding_obj,
            persist_directory=persist_directory
        )
        return db, embedding_fn
    except Exception as e:
        logging.warning(f"Failed to initialize Chroma DB: {e}. Attempting reinit if dimension mismatch.")
        msg = str(e).lower()
        if "dimension" in msg or "embedding" in msg or "mismatch" in msg:
            _reinit_chroma_dir(persist_directory)
            try:
                db = Chroma(
                    collection_name="rag_docs",
                    embedding_function=embedding_obj,
                    persist_directory=persist_directory
                )
                return db, embedding_fn
            except Exception as e2:
                logging.warning(f"Reinit retry failed: {e2}")
        return None, embedding_fn

def add_documents_to_chroma(texts, metadatas=None, persist_directory=CHROMA_DB_DIR):
    """
    Add texts to Chroma if available; otherwise append to fallback JSON store.
    On dimension/shape errors, reinitialize DB and retry once.
    """
    db, embedding_fn = get_chroma_db(persist_directory=persist_directory)
    if db is not None:
        try:
            db.add_texts(texts=texts, metadatas=metadatas or [{} for _ in texts])
            try:
                db.persist()
            except Exception:
                pass
            logging.info(f"Added {len(texts)} documents to ChromaDB at {persist_directory}.")
            return
        except Exception as e:
            logging.warning(f"Chroma add_texts failed: {e}")
            msg = str(e).lower()
            if "dimension" in msg or "mismatch" in msg or "shape" in msg:
                logging.info("Detected embedding dimension mismatch — reinitializing Chroma DB and retrying.")
                _reinit_chroma_dir(persist_directory)
                db, _ = get_chroma_db(persist_directory=persist_directory)
                try:
                    if db is not None:
                        db.add_texts(texts=texts, metadatas=metadatas or [{} for _ in texts])
                        try:
                            db.persist()
                        except Exception:
                            pass
                        logging.info(f"Added {len(texts)} documents to ChromaDB after reinit.")
                        return
                except Exception as e2:
                    logging.warning(f"Retry after reinit failed: {e2}")

            logging.warning("Writing to fallback store after Chroma failure.")

    # Fallback JSON store
    items = []
    if FALLBACK_STORE.exists():
        try:
            items = json.loads(FALLBACK_STORE.read_text(encoding="utf-8"))
        except Exception:
            items = []
    embs = embedding_fn(texts)
    for i, t in enumerate(texts):
        items.append({
            "id": f"doc-{len(items)+1}",
            "text": t,
            "metadata": (metadatas[i] if metadatas and i < len(metadatas) else {}),
            "embedding": embs[i]
        })
    FALLBACK_STORE.write_text(json.dumps(items, indent=2), encoding="utf-8")
    logging.info(f"Wrote {len(texts)} documents to fallback store: {FALLBACK_STORE}")

def search_chroma(query, k=3, persist_directory=CHROMA_DB_DIR):
    db, embedding_fn = get_chroma_db(persist_directory=persist_directory)
    if db is not None:
        try:
            return db.similarity_search(query, k=k)
        except Exception as e:
            logging.warning(f"Chroma search failed: {e} — falling back to JSON store.")

    if not FALLBACK_STORE.exists():
        logging.info("No fallback store found.")
        return []
    data = json.loads(FALLBACK_STORE.read_text(encoding="utf-8"))
    qtokens = set(query.lower().split())
    scored = []
    for d in data:
        score = len(qtokens & set(d.get("text", "").lower().split()))
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:k]]
