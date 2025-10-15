import streamlit as st
from scripts.web_crawler import crawl_webpage
from scripts.pdf_scraper import extract_sections_from_pdf
from scripts.chromadb_setup import add_documents_to_chroma
from scripts.rag_pipeline import run_query
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

st.set_page_config(page_title="RAG System with Groq LLM & LangChain", layout="wide")

st.title("Retrieval-Augmented Generation (RAG) System")

# Sidebar navigation
step = st.sidebar.selectbox(
    "Select Step",
    ("Web Crawl", "PDF Scraper", "Query", "Full Pipeline Demo")
)

if step == "Web Crawl":
    st.header("Step 1: Web Crawler")
    url = st.text_input("Enter a URL to crawl", value="https://example.com")
    if st.button("Crawl and Add to Vector DB"):
        if not url:
            st.error("Please enter a valid URL.")
        else:
            try:
                with st.spinner("Crawling webpage and adding to ChromaDB..."):
                    text = crawl_webpage(url)
                    add_documents_to_chroma([text], [{"source": url}])
                st.success("Webpage content added to vector database successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

elif step == "PDF Scraper":
    st.header("Step 2: PDF Scraper")
    pdf_file = st.file_uploader("Upload a research paper PDF", type=["pdf"])
    if st.button("Extract and Add to Vector DB"):
        if not pdf_file:
            st.error("Please upload a PDF file.")
        else:
            try:
                with st.spinner("Extracting sections from PDF and adding to ChromaDB..."):
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.read())
                        tmp_path = tmp.name
                    sections = extract_sections_from_pdf(tmp_path)
                    texts = [v for v in sections.values() if v]
                    metadatas = [{"source": f"{pdf_file.name} - {k}"} for k in sections.keys()]
                    if texts:
                        add_documents_to_chroma(texts, metadatas)
                        st.success(f"Extracted and added sections from PDF: {', '.join(sections.keys())}")

                        # Display extracted sections on UI
                        st.subheader("Extracted Sections Preview")
                        for section_name, section_text in sections.items():
                            if section_text:
                                with st.expander(f"{section_name}"):
                                    st.write(section_text)
                    else:
                        st.warning("No text extracted from the PDF.")
            except Exception as e:
                st.error(f"Error: {e}")


elif step == "Query":
    st.header("Step 3: Query RAG System")
    query = st.text_area("Enter your query", value="What are the latest methods for improving transformer architectures in NLP?")
    if st.button("Run Query"):
        if not query.strip():
            st.error("Please enter a query.")
        else:
            try:
                with st.spinner("Running query through RAG pipeline..."):
                    result = run_query(query)
                st.subheader("Answer")
                st.write(result.get('result'))
                st.subheader("Sources")
                for doc in result.get('source_documents', []):
                    src = "unknown"
                    try:
                        src = doc.get("metadata", {}).get("source", "unknown")
                    except Exception:
                        try:
                            src = getattr(doc, "metadata", {}).get("source", "unknown")
                        except Exception:
                            src = str(doc)
                    st.markdown(f"- {src}")
            except Exception as e:
                st.error(f"Error: {e}")

elif step == "Full Pipeline Demo":
    st.header("Full Pipeline Demo (Web Crawl -> Add -> Query)")
    sample_url = st.text_input("Sample URL", value="https://example.com")
    demo_query = st.text_area("Demo Query", value="What is retrieval augmented generation?")
    if st.button("Run Full Demo"):
        if not sample_url or not demo_query.strip():
            st.error("Please provide both sample URL and query.")
        else:
            try:
                with st.spinner("Running full pipeline..."):
                    text = crawl_webpage(sample_url)
                    add_documents_to_chroma([text], [{"source": sample_url}])
                    result = run_query(demo_query)
                st.subheader("Answer")
                st.write(result.get('result'))
                st.subheader("Sources")
                for doc in result.get('source_documents', []):
                    src = "unknown"
                    try:
                        src = doc.get("metadata", {}).get("source", "unknown")
                    except Exception:
                        try:
                            src = getattr(doc, "metadata", {}).get("source", "unknown")
                        except Exception:
                            src = str(doc)
                    st.markdown(f"- {src}")
            except Exception as e:
                st.error(f"Error: {e}")
