import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
def ingest_pdf(file_path: str, index_name: str):
    print(f"Starting ingestion for {file_path} into index {index_name}...")

    print("Loading PDF document...")
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages from the document.")
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Document split into {len(docs)} chunks.")
    print("Creating embeddings... (This may take a while for large documents)")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print("Embeddings model loaded.")
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating new serverless index...")
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension for 'all-MiniLM-L6-v2'
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists. Proceeding to add documents.")

    print(f"Storing document embeddings in Pinecone index '{index_name}'...")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    print("=" * 30)
    print("✅ Ingestion Complete ✅")
    print(f"Your PDF has been processed and stored in the Pinecone index '{index_name}'.")
    print("You can now run the chatbot application.")
    print("=" * 30)