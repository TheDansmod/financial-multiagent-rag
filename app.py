import os
import chainlit as cl
from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

# --- Configuration ---
# Must match the ingestion script
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"

# --- Helper: Format Docs ---
# This replaces "create_stuff_documents_chain".
# It simply joins document content with newlines.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 1. Initialize Components on Chat Start ---
@cl.on_chat_start
async def on_chat_start():
    try:
        # A. Setup Embeddings (Same as ingest)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # B. Connect to Qdrant (Load the existing DB)
        if not os.path.exists(QDRANT_PATH):
            await cl.Message(content=f"Error: Qdrant DB not found at {QDRANT_PATH}").send()
            return
        client = QdrantClient(path=QDRANT_PATH)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        
        # C. Create Retriever
        # search_kwargs={"k": 2} means we retrieve the top 2 most relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        # D. Setup LLM (Ollama)
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0, # Keep it factual
        )

        # E. Create Prompt Template
        # This tells the LLM: "Use the provided context to answer the question."
        # the {context} and {input} are variables whose value will be provided later
        template = (
            "You are a helpful assistant. Use the following pieces of retrieved context "
            "to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\n"
            "Context:\n"
            "{context}\n\n"
            "Question:\n"
            "{question}"
        )
        prompt = ChatPromptTemplate.from_template(template)

        # E. Build the Pure LCEL Chain
        # Step 1: Retrieval Step
        # This parallel branch retrieves docs and passes the original question through.
        # We use a dict so we can return 'context' (docs) to the UI later.
        setup_and_retrieval = RunnableParallel(
            {
                "context": itemgetter("input") | retriever,
                "question": itemgetter("input"),
            }
        )

        # Step 2: Generation Step
        # We take the output of Step 1, format the context, and pass it to the LLM.
        # We use .assign() to add the "answer" key to our dictionary, preserving the "context".
        rag_chain = setup_and_retrieval.assign(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: format_docs(x["context"]) # Format docs into string here
                )
                | prompt
                | llm
                | StrOutputParser()
            )
        )

        cl.user_session.set("rag_chain", rag_chain)
        await cl.Message(content="Hello! I am ready to chat about your documents.").send()

    except Exception as e:
        await cl.Message(content=f"Failed to initialize: {str(e)}").send()


# --- 2. Handle Messages ---
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from session
    rag_chain = cl.user_session.get("rag_chain")

    # Create a placeholder message (this shows the "Thinking..." animation)
    msg = cl.Message(content="")

    await msg.send()

    source_documents = []
    # Run the RAG chain
    # The chain expects "input" as the user's question
    async for chunk in rag_chain.astream({"input": message.content}):
        # We are interested in the "answer" key from the result
        if "answer" in chunk:
            await msg.stream_token(chunk["answer"])
        # 'context' contains the list of documents (yielded once)
        if "context" in chunk:
            source_documents = chunk["context"]

    # Finalize the message
    if source_documents:
        text_elements = []
        for idx, doc in enumerate(source_documents):
            source_name = doc.metadata.get("source", f"Source {idx+1}")
            text_elements.append(
                cl.Text(content=doc.page_content, name=source_name, display="inline")
            )
        msg.elements = text_elements

    await msg.update()
