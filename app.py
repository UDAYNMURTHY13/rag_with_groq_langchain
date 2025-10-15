import streamlit as st
from scripts.web_crawler import crawl_webpage
from scripts.pdf_scraper import extract_sections_from_pdf
from scripts.chromadb_setup import add_documents_to_chroma
from scripts.rag_pipeline import run_query

st.set_page_config(page_title="RAG System UI", layout="wide")
st.title("🔍 RAG System Interface")

# Sidebar for selecting step
step = st.sidebar.selectbox(
    "Select pipeline step to execute",
    options=["web", "pdf", "query", "all"],
    index=2,
)

if step == "all":
    st.info("Running full pipeline demo (web -> add -> query)")
    sample_url = "https://example.com"
    if st.button("Run Full Pipeline Demo"):


        with st.spinner("Crawling webpage..."):
            text = crawl_webpage(sample_url)

        with st.spinner("Adding document to ChromaDB..."):
            add_documents_to_chroma([text], [{"source": sample_url}])
        st.success("Full pipeline demo finished!")

elif step == "web":
    url = st.text_input("Enter webpage URL", "")
    if st.button("Crawl & Add Webpage"):
        if not url:
            st.error("Please enter a URL")
        else:
            try:
                with st.spinner(f"Crawling {url}..."):
                    text = crawl_webpage(url)
                with st.spinner("Adding document to ChromaDB..."):
                    add_documents_to_chroma([text], [{"source": url}])
                st.success(f"Webpage content added for URL: {url}")
            except Exception as e:
                st.error(f"Error: {e}")

elif step == "pdf":
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
    if st.button("Extract & Add PDF"):
        if not pdf_file:
            st.error("Please upload a PDF file")
        else:
            try:
                # Save uploaded PDF temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(pdf_file.read())
                with st.spinner("Extracting text from PDF..."):
                    sections = extract_sections_from_pdf("temp.pdf")
                texts = [v for v in sections.values() if v]
                metadatas = [{"source": f"{pdf_file.name} - {k}"} for k in sections.keys()]
                if texts:
                    with st.spinner("Adding documents to ChromaDB..."):
                        add_documents_to_chroma(texts, metadatas)
                    st.success(f"PDF content added: {pdf_file.name}")
                else:
                    st.warning("No text extracted from PDF.")
            except Exception as e:
                st.error(f"Error: {e}")

elif step == "query":
    query = st.text_area("Enter your query", height=100)
    if st.button("Run Query"):
        if not query.strip():
            st.error("Please enter a query.")
        else:
            try:
                with st.spinner("Running query through RAG pipeline..."):
                    result = run_query(query)
                answer = result.get("result", "No answer found.")
                sources = result.get("source_documents", [])

                st.subheader("📝 Answer")
                st.write(answer)

                st.subheader("📚 Sources")
                if sources:
                    for i, doc in enumerate(sources, 1):
                        src = doc.get("metadata", {}).get("source", "Unknown source")
                        st.write(f"{i}. {src}")
                else:
                    st.write("No sources found.")
            except Exception as e:
                st.error(f"Error running query: {e}")


