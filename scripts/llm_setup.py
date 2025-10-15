import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
logging.info(f"GROQ_API_KEY present: {bool(GROQ_API_KEY)}; MODEL_NAME: {MODEL_NAME}")

# Try to import concrete provider (langchain_groq.ChatGroq)
Provider = None
try:
    from langchain_groq import ChatGroq
    Provider = ChatGroq
    logging.info("Using langchain_groq.ChatGroq provider.")
except Exception as e:
    logging.debug(f"langchain_groq.ChatGroq import failed: {e}")
    Provider = None

class _StubLLM:
    def __init__(self, model="stub", temperature=0.0):
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt: str, **kwargs):
        head = (prompt or "")[:400].replace("\n", " ")
        return f"[LLM_STUB] model={self.model} temp={self.temperature} | prompt_preview={head}"

class GroqWrapper:
    """Wraps a ChatGroq-like provider and exposes a simple callable returning text."""
    def __init__(self, provider):
        self._p = provider

    def __call__(self, prompt: str, **kwargs):
        try:
            if hasattr(self._p, "invoke"):
                resp = self._p.invoke(prompt)
                return getattr(resp, "content", getattr(resp, "text", str(resp)))
            if callable(self._p):
                out = self._p(prompt)
                return out if isinstance(out, str) else str(out)
            if hasattr(self._p, "generate"):
                g = self._p.generate([prompt])
                try:
                    return g.generations[0][0].text
                except Exception:
                    return str(g)
        except Exception as e:
            logging.warning(f"Provider call failed: {e}")
            return f"[LLM_ERROR] {e}"
        return "[LLM_ERROR] Unsupported provider shape."

def get_llm(model_name: str = MODEL_NAME, temperature: float = 0.0):
    """
    Return a callable LLM. Prefers ChatGroq provider (langchain_groq) when available
    and GROQ_API_KEY is set. Otherwise returns a local stub for offline use.
    """
    if Provider and GROQ_API_KEY:
        try:
            try:
                inst = Provider(model=model_name, temperature=temperature, api_key=GROQ_API_KEY)
            except TypeError:
                try:
                    inst = Provider(model=model_name, temperature=temperature)
                except TypeError:
                    inst = Provider(model_name)
            logging.info("Initialized ChatGroq provider.")
            return GroqWrapper(inst)
        except Exception as e:
            logging.warning(f"Failed to instantiate ChatGroq provider: {e}. Falling back to stub.")
            return _StubLLM(model=model_name, temperature=temperature)
    logging.info("ChatGroq provider unavailable or GROQ_API_KEY missing — returning stub LLM.")
    return _StubLLM(model=model_name, temperature=temperature)

if __name__ == "__main__":
    llm = get_llm()
    prompt = "Write one line about what Groq LLM does."
    print("Prompt ->", prompt)
    print("Response ->", llm(prompt))