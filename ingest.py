import os
import sys
from uuid import uuid4
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# configuration
DATA_PATH = "./data/my_knowledge.txt"
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Load the data
print("Loading data...")
if not os.path.exists(DATA_PATH):
    print(f"Error: File not found at {DATA_PATH}")
    sys.exit(1)
loader = TextLoader(DATA_PATH)
documents = loader.load()

# 2. Split text into chunks
# We split the text because LLMs have context limits and we want to find specific details.
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Initialize Embeddings
# We use a standard lightweight HuggingFace model.
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 4. Initialize Qdrant and store vectors
# We use local mode (path="./qdrant_db") so you don't need to run a Docker container.
print("Indexing data into Qdrant...")

# Ensure the collection exists (optional but good practice for persistent stores)
client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=client.get_embedding_size(EMBEDDING_MODEL), distance=Distance.COSINE),
    )
else:
    print(f"Collection '{COLLECTION_NAME}' already exists. Appending documents...")

# Create the Vector Store
# qdrant = QdrantVectorStore.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     client=client,
#     collection_name=COLLECTION_NAME,
# )
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
vector_store.add_documents(documents=docs, ids=[str(uuid4()) for _ in range(len(docs))])

print(f"Success! {len(docs)} chunks indexed.")
