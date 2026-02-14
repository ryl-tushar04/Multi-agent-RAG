from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec
from config import *
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from config import OPENAI_API_KEY

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)


def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )

    return pc.Index(INDEX_NAME)



def ingest_documents(data_path="./docs"):
    index = init_pinecone()
    vector_store = PineconeVectorStore(pinecone_index=index)

    documents = SimpleDirectoryReader(data_path).load_data()

    storage_index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
    )

    return storage_index