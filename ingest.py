import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# 1. Load the data
print("Loading data...")
loader = TextLoader("./data/my_knowledge.txt")
documents = loader.load()

# 2. Split text into chunks
# We split the text because LLMs have context limits and we want to find specific details.
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Initialize Embeddings
# We use a standard lightweight HuggingFace model.
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Initialize Qdrant and store vectors
# We use local mode (path="./qdrant_db") so you don't need to run a Docker container.
print("Indexing data into Qdrant...")

url = "./qdrant_db"  # Local persistence

# Ensure the collection exists (optional but good practice for persistent stores)
client = QdrantClient(path=url)
collection_name = "my_documents"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Create the Vector Store
qdrant = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    url=url,
    collection_name=collection_name,
)

print(f"Success! {len(docs)} chunks indexed.")
