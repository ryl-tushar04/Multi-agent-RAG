
import streamlit as st
import os
from ingestion import ingest_documents
from agents import create_agent

st.set_page_config(page_title="Multi-Agent RAG", layout="wide")

st.title("🧠 Multi-Agent RAG System")
st.caption("Llama3.2 (Ollama) + LlamaIndex + Pinecone + Tavily")

# ---------- Sidebar ----------
st.sidebar.header("📂 Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

DOCS_DIR = "./docs"
os.makedirs(DOCS_DIR, exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DOCS_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.sidebar.success("Files uploaded successfully")

    if st.sidebar.button("🔄 Re-index Documents"):
        with st.spinner("Indexing documents into Pinecone..."):
            ingest_documents(DOCS_DIR)
        st.sidebar.success("Indexing completed")

# ---------- Agent Init ----------
if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        st.session_state.agent = create_agent()

agent = st.session_state.agent

# ---------- Chat UI ----------
st.subheader("💬 Ask Questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your question")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        response = agent.invoke({"input": query})
        answer = response["output"]

    st.session_state.chat_history.append((query, answer))

# ---------- Display Chat ----------
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
    st.markdown("---")

# ---------- Footer ----------
st.caption("Powered by Multi-Agent RAG Architecture")