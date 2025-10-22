# RAG with Groq & LangChain

A **Retrieval-Augmented Generation (RAG)** system leveraging **LangChain**, **ChromaDB**, and **Groq** for fast and accurate question-answering over custom documents.

---

## 🔹 Overview

This project demonstrates a pipeline that:

1. Retrieves relevant documents from a vector database.
2. Uses a language model to generate context-aware responses.
3. Integrates **Groq** hardware for efficient inference.

Applications include:
- Knowledge base assistants
- Document Q&A systems
- AI-powered chatbots

---

## 🗂️ Project Structure

```

rag\_with\_groq\_langchain/
│
├── app.py                 \# Main application script
├── streamlit\_app.py       \# Streamlit UI for RAG
├── requirements.txt       \# Python dependencies
├── data/                  \# Folder for documents / datasets
├── chroma\_db/             \# Chroma vector database storage
├── scripts/               \# Utility scripts (embedding, preprocessing)

````

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone [https://github.com/UDAYNMURTHY13/rag_with_groq_langchain.git](https://github.com/UDAYNMURTHY13/rag_with_groq_langchain.git)
cd rag_with_groq_langchain
````

2.  **Create a virtual environment**

<!-- end list -->

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3.  **Install dependencies**

<!-- end list -->

```bash
pip install -r requirements.txt
```

-----

## 🚀 Usage

**Run the main app (console)**

```bash
python app.py
```

**Run Streamlit interface**

```bash
streamlit run streamlit_app.py
```

-----

## 🧠 How It Works

1.  **Document Embedding**: Converts documents in `data/` into vector embeddings using **ChromaDB**.
2.  **Retrieval**: When a user asks a query, the system fetches the most relevant documents (chunks).
3.  **Generation**: The language model (**Groq**-backed) generates an answer using the retrieved context.
4.  **Output**: Response is returned to the user via console or Streamlit interface.

-----

## 🔧 Dependencies

Key Python packages:

  * `langchain`
  * `chromadb`
  * `streamlit`
  * `openai` (or any LLM backend configured with LangChain)

Other dependencies listed in `requirements.txt`.

-----

## 📚 References

  * [LangChain Documentation](https://www.langchain.com/)
  * [ChromaDB](https://www.trychroma.com/)
  * [Groq AI](https://groq.com/)

-----

## ✨ Contribution

Feel free to fork, raise issues, or submit pull requests.

-----

## 📝 License

This project is licensed under the **MIT License**.


