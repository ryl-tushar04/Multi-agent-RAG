import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
import rag_logic

load_dotenv()

st.set_page_config(page_title="PDF Chatbot with RAG Logic", layout="wide")
st.title("📄 Chat with Your PDF")

INDEX_NAME = "pdf-chatbot-index"

@st.cache_resource
def get_models():
    llm = Ollama(model='llama3', temperature=0.2)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return llm, embeddings

llm, embeddings = get_models()

with st.sidebar:
    st.header("Upload Your PDF")
    st.info(
        "Upload your document and click 'Process' to get it ready for questions. "
        "**Note:** This will replace any previously processed document in the index."
    )
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document... This may take a moment."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    rag_logic.ingest_pdf(file_path=tmp_file_path, index_name=INDEX_NAME)
                    st.success("Document processed successfully!")
                    st.session_state.processed = True
                except Exception as e:
                    st.error(f"Failed to process the document. Error: {e}")
                    st.session_state.processed = False
                finally:
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
        else:
            st.warning("Please upload a PDF file first.")
    st.markdown("---")
    st.markdown("Created by Tushar Saxena")
if "processed" in st.session_state and st.session_state.processed:
    st.info("Your document has been processed. You can now ask questions.")

    try:
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

        prompt_template = """
        You are an intelligent assistant that answers questions based on the provided context from a PDF document.
        Use only the following context to answer the question. If you don't know the answer from the context, just say that you don't know. Do not try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! Ask me anything about your document."}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your question..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                source_docs = response["source_documents"]

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("View Sources"):
                        for i, doc in enumerate(source_docs):
                            st.write(f"**Source {i + 1} (Page {doc.metadata.get('page', 'N/A')})**")
                            st.info(doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(
            f"Failed to initialize the QA chain. Please ensure your Pinecone index is set up correctly. Error: {e}")
else:
    st.warning("Please upload and process a PDF using the sidebar to begin chatting.")