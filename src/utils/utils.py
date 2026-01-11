"""This function will contain several utility functions useful for the project."""

import json
import logging
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

log = logging.getLogger(__name__)


def get_llm_provider(name, *args, **kwargs):
    """Get the right LLM provider based on the model name."""
    if name in ["llama3.1", "deepseek-r1:8b"]:
        return ChatOllama
    elif "mistral" in name or "mixtral" in name:
        return ChatMistralAI
    elif "gemini" in name or "gemma" in name:
        return ChatGoogleGenerativeAI


def append_to_json_list_file(file_path: str, entry):
    """Append a single record to the JSON list file.

    This is inefficient (Read-Modify-Write) but is robust to crashes and gives some
    time between successive LLM calls to prevent exceeding the rate limit
    """
    path = Path(file_path)
    data = load_from_json_list_file(path, fail_on_error=False)
    data.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_from_json_list_file(path: Path | str, fail_on_error=False):
    """Load JSON content from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if fail_on_error:
            log.error(f"Error while loading json: {repr(e)}.\nFailing.")
            raise Exception(f"Error while loading json file with path {path}")
        else:
            log.error(f"Error while loading json: {repr(e)}\nContinuing anyway.")
            return []

def get_newline_split_markdown(path):
    """Read markdown file and split by double newlines, trimming whitespace."""
    with open(path, "r", encoding="utf-8") as f:
        # split by double newline to separate paragraphs/tables
        splits = f.read().split("\n\n")
    return [s.strip() for s in splits]

