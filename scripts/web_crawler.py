import logging
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(message)s")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "web_content"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_web_content(url: str) -> str:
    """
    Fetch plain text content from a webpage, save to data/web_content/<host>.txt,
    and return the extracted text.
    """
    try:
        logging.info(f"Fetching: {url}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for elem in soup(["script", "style", "header", "footer", "nav", "aside"]):
            elem.extract()
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])
        if not text:
            text = soup.get_text(separator="\n")
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        host = urlparse(url).netloc or "page"
        out_file = OUT_DIR / f"{host}.txt"
        out_file.write_text(text, encoding="utf-8")
        logging.info(f"Saved webpage text -> {out_file}")
        return text
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return ""
    
def crawl_webpage(url: str) -> str:
    return fetch_web_content(url)

if __name__ == "__main__":
    test = "https://zilliz.com/tutorials/rag/langchain-and-langchain-vector-store-and-groq-llama3-70b-8192-and-voyage-3"
    print(fetch_web_content(test)[:1000])