"""Global Doc String."""

import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def get_sec_10k_data(data_folder, companies):
    """Doc string."""
    # data_folder = "/mnt/windows/Users/lordh/Documents/Svalbard/Data"
    # companies = cfg.data.companies
    from sec_edgar_downloader import Downloader

    dl = Downloader("Google", "vke4@gmail.com", data_folder)
    for company in companies:
        try:
            dl.get("10-K", company, limit=1, download_details=False)
            print(f"Downloaded 10-K for {company}")
        except Exception as e:
            print(f"Failed to download {company}: {e}")


def clean_and_convert_to_markdown(cfg):
    """Doc string."""
    import re

    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    data_folder = r"data/sec_filings/"
    for company in cfg.data.companies:
        with (
            open(f"{data_folder}{company}-full-submission.txt", "r") as fr,
            open(f"{data_folder}{company}-parsed.md", "w") as fw,
        ):
            raw_content = fr.read()
            soup = BeautifulSoup(raw_content, "html.parser")
            for script in soup(["script", "style", "head", "title", "meta"]):
                script.extract()

            # 3. Convert to Markdown (Preserves tables & headers)
            markdown_text = md(str(soup), heading_style="ATX")

            # 4. Clean up excessive newlines created by conversion
            markdown_text = re.sub(r"\n\s*\n", "\n\n", markdown_text)
            fw.write(markdown_text)


def convert_sec_filing_to_markdown(src_filepath, dst_filepath):
    """Doc string."""
    import re

    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    with open(src_filepath, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # 1. Regex to find all <DOCUMENT> blocks
    # SEC files structure:
    # <DOCUMENT>\n<TYPE>10-K...\n<TEXT>...html content...</TEXT>\n</DOCUMENT>
    doc_start_pattern = re.compile(r"<DOCUMENT>")
    doc_end_pattern = re.compile(r"</DOCUMENT>")

    type_pattern = re.compile(r"<TYPE>(.*?)[\n\r]")

    # Find start and end positions of all documents
    starts = [m.start() for m in doc_start_pattern.finditer(raw_content)]
    ends = [m.end() for m in doc_end_pattern.finditer(raw_content)]

    target_document = None

    for start, end in zip(starts, ends):
        doc_content = raw_content[start:end]

        # Check the document TYPE
        type_match = type_pattern.search(doc_content)
        if type_match:
            doc_type = type_match.group(1).strip()

            # We only want main 10-K filing (sometimes labeled 10-K/A for amendments)
            if doc_type == "10-K":
                target_document = doc_content
                break

    if not target_document:
        return "Error: 10-K document not found in the file."

    # 2. Extract the HTML content specifically between <TEXT> tags
    text_content_match = re.search(
        r"<TEXT>(.*?)</TEXT>", target_document, re.DOTALL
    )  # matches newline as well
    if not text_content_match:
        return "Error: No <TEXT> content found in the 10-K document."

    html_content = text_content_match.group(1)

    # 3. Use BeautifulSoup to handle the specific HTML (Cleaning)
    # This prevents the 'zip' error because we excluded the binary documents
    soup = BeautifulSoup(html_content, "lxml")  # lxml is faster for large files

    # Optional: Remove XBRL tags or hidden tables if they clutter the markdown
    for tag in soup.find_all(["xml", "type"]):
        tag.decompose()

    # 4. Convert to Markdown
    # strip=['a'] removes links to keep it clean for RAG, keep them if you need sources
    markdown_text = md(str(soup), heading_style="ATX", strip=["img", "a"])

    # Post-processing to clean up excessive newlines common in SEC conversions
    markdown_text = re.sub(r"\n\s*\n", "\n\n", markdown_text)

    with open(dst_filepath, "w") as f:
        f.write(markdown_text)


def get_markdown_sec_filings(ticker, dest_filepath):
    """Doc string."""
    # data_folder = r"data/edgar_tools_filings/"
    # for company in cfg.data.companies:
    #     print(company)
    #     src_filepath = f"{data_folder}{company}-full-submission.txt"
    #     dst_filepath = f"{data_folder}{company}.md"
    #     get_markdown_sec_filings(company, dst_filepath)
    from edgar import Company, set_identity

    set_identity("Vipin Kumar vipinkumar1993@gmail.com")
    company = Company(ticker)
    filing = company.get_filings(form="10-K").latest()
    text = filing.markdown()
    with open(dest_filepath, "w") as f:
        f.write(text)


def clean_edgar_markdown(src_filepath, dst_filepath):
    """Doc string."""
    # data_folder = r"data/edgar_tools_filings/"
    # for company in cfg.data.companies:
    #     print(company)
    #     src_filepath = f"{data_folder}{company}.md"
    #     dst_filepath = f"{data_folder}{company}-cleaned.md"
    #     clean_edgar_markdown(src_filepath, dst_filepath)
    import re

    with open(src_filepath, "r") as f:
        content = f.read()
    pattern = re.compile(r"<div[^>]*>|</div>", flags=re.IGNORECASE)
    cleaned_content = pattern.sub("", content)
    with open(dst_filepath, "w") as f:
        f.write(cleaned_content)


def generate_qdrant_db(cfg):
    """Doc string."""
    # with hydra.initialize(version_base=None, config_path="."):
    #     cfg: DictConfig = hydra.compose(
    #       config_name="config", overrides=[], return_hydra_config=True)
    # hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    # generate_qdrant_db(cfg)
    from uuid import uuid4

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    # get the markdown splitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # get embedding for the splits and create/get the collection
    embeddings = HuggingFaceEmbeddings(model_name=cfg.vector_db.embedding_model)
    client = QdrantClient(path=cfg.vector_db.qdrant_path)
    if not client.collection_exists(cfg.vector_db.collection_name):
        log.debug(f"Creating collection '{cfg.vector_db.collection_name}'...")
        client.create_collection(
            collection_name=cfg.vector_db.collection_name,
            vectors_config=VectorParams(
                size=client.get_embedding_size(cfg.vector_db.embedding_model),
                distance=Distance.COSINE,
            ),
        )
    else:
        log.debug(
            f"Collection '{cfg.vector_db.collection_name}' already exists."
            " Appending documents..."
        )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=cfg.vector_db.collection_name,
        embedding=embeddings,
    )

    for idx, company in enumerate(cfg.data.companies):
        log.debug(f"generating split documents for company no. {idx + 1}: {company}")
        # get the text from the cleaned markdown file
        with open(f"{cfg.data.data_folder}/{company}-cleaned.md", "r") as f:
            text = f.read()
        # split it using the recursive splitter
        header_splits = md_splitter.split_text(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.vector_db.chunk_size,
            chunk_overlap=cfg.vector_db.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = text_splitter.split_documents(header_splits)
        # add metadata to splits
        for split in splits:
            split.metadata["company"] = company
            split.metadata["year"] = cfg.data.year
        # add the splits to the vector db
        log.debug(f"Adding {len(splits)} documents for company {company}")
        vector_store.add_documents(
            documents=splits, ids=[str(uuid4()) for _ in range(len(splits))]
        )


def test_llama_json_handling(model_name: str, model_temp: float):
    """Test if input model handles complex JSON queries properly.

    Quantized models of smaller size - like the one I am using in Llama 3.1 8B -
       often suffer from "Context Collapse" or lose the ability to output valid
       JSON when the prompt is complex. Since my Supervisor relies on JSON output
       to route queries, this is a critical failure point. This function is
       intended to evaluate how the model performs when tasked with providing JSON
       output to a complex input prompt. It prints the content of the response
       obtained from the LLM to a prompt that asks the model to output only in json.

    Args:
        model_name (str): The name of the Ollama model which should be invoked
        model_temp (float): The temperature at which to use the model. The
            temperature controls the token selection. If the temperature is 0 then
            only the highest probability token is always selected - essentially
            introducing determinism into the model. If the temperature is higher
            the likelihood of selecting a token other than the highest probability
            token is increased.

    Returns:
        None: The function has no return value. It only prints the content of the
            response.
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model_name, temperature=model_temp)
    prompt = """
You are a router. You must output JSON only.
Analyze the user query: "Compare the R&D spending of Apple and Microsoft in 2023."
Return a JSON object with this schema:
{
  "intent": "comparison",
  "entities": ["list", "of", "companies"],
  "metric": "financial_metric_extracted"
}
Do not output any conversational text.
"""
    response = llm.invoke(prompt)
    log.info(f"Response:\n{response.content}")


if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path="."):
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    test_llama_json_handling(cfg.model.name, cfg.model.temp)
