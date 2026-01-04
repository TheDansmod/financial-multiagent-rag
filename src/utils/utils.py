from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm_provider(name: str, *args, **kwargs):
    if name in ["llama3.1", "deepseek-r1:8b"]:
        return ChatOllama
    elif "mistral" in name or "mixtral" in name:
        return ChatMistralAI
    elif "gemini" in name or "gemma" in name:
        return ChatGoogleGenerativeAI

