"""Exploring various things in the RAG - they are tested out here first.

In this file we will be trying out various things to see how they work, or we will be
exploring the datasets and techniques etc.

We are using the following queries for testing:
    fact_query = "Who is the auditor for Apple in 2023?"
    comparison_query = "Compare the risk factors for apple and microsoft."
    complex_query = (
        "What is the primary risk factor listed by the company"
        "with the highest revenue in the tech sector?"
    )

"""

import logging
from typing import Literal, TypedDict

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()
log = logging.getLogger(__name__)


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


def explore_vector_db(cfg):
    from qdrant_client import QdrantClient

    qdrant_path, coll_name = cfg.vector_db.qdrant_path, cfg.vector_db.collection_name
    client = QdrantClient(path=qdrant_path)
    points = client.scroll(
        collection_name=coll_name,
        limit=1,
        with_payload=["metadata"],
        with_vectors=False,
    )
    # points[0] will have length = limit, each element is a Record, each record
    # has record.payload, each payload is a dictionary which has key metadata.
    # The value of the metadata key is also a dictionary with contains Header N
    # (maybe only some N), company, and year.
    print(points)


class QueryType(TypedDict):
    query_type: Literal["simple", "comparison", "complex"]
    query_companies: (
        list[
            Literal[
                "NVDA",
                "AAPL",
                "MSFT",
                "AMZN",
                "GOOGL",
                "META",
                "TSLA",
                "WMT",
                "JPM",
                "NFLX",
            ]
        ]
        | None
    )


class SubQueries(TypedDict):
    sub_queries: list[str]
    company: list[
        Literal[
            "NVDA",
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "WMT",
            "JPM",
            "NFLX",
        ]
    ]


def test_supervisor(cfg, user_query):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    log.debug(f"query is: {user_query}")
    model = ChatOllama(model=cfg.model.name, temperature=cfg.model.temp)
    template = cfg.agent_configs.supervisor_node.system_prompt_template
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(user_query=user_query)
    structured_llm = model.with_structured_output(QueryType)
    classification = structured_llm.invoke(formatted_prompt)
    log.info(classification)


def test_planner(cfg, user_query):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI as Gemini

    log.debug(f"query is: {user_query}")
    # model = ChatOllama(model=cfg.model.name, temperature=cfg.model.temp)
    model = Gemini(model=cfg.model.name, temperature=cfg.model.temp)
    template = cfg.agent_configs.planner_node.system_prompt_template
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(user_query=user_query)
    log.debug(f"Formatted prompt is: {formatted_prompt}")
    structured_llm = model.with_structured_output(SubQueries)
    main_sub_query = structured_llm.invoke(formatted_prompt)
    log.info(main_sub_query)


def test_retriever(cfg, sub_query):
    from langchain_huggingface import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient

    qdrant_path, coll_name = cfg.vector_db.qdrant_path, cfg.vector_db.collection_name
    num_fetch_points = cfg.agent_configs.retriever_node.num_closest_chunks
    embeddings = HuggingFaceEmbeddings(model_name=cfg.vector_db.embedding_model)
    query_vector = embeddings.embed_query(sub_query)
    client = QdrantClient(path=qdrant_path)
    matched_points = client.query_points(
        collection_name=coll_name,
        query=query_vector,
        limit=num_fetch_points,
        with_payload=["metadata", "page_content"],
        with_vectors=False,
    )
    log.info(matched_points)


def test_mistral(cfg):
    from hydra.utils import instantiate

    provider = instantiate(cfg.model)
    model = provider(model=cfg.model.name, temperature=cfg.model.temp)
    prompt = (
        "When using the oil plugin on neovim, I have added the "
        "option show_hidden = true in the view_options section for the plugin, "
        "but when I close neovim and re-open it, the hidden files are still"
        " not visible. How to fix this issue?"
    )
    response = model.invoke(prompt)
    log.info(response)
    log.info(f"\n\n{response.content}\n\n")


def analyse_headers(cfg):
    r"""Figure out what all the headers are like, for all the files you have.

    We have figured out the following so far:
    1. Number of points in the collection = 7687
       ```num_points = client.count(collection_name=coll_name, count_filter=None)```
    2. Number of points per company:
       ```
       for ticker in cfg.data.companies:
           query_filter = Filter(
               must=[
                   FieldCondition(
                       key="metadata.company", match=MatchValue(value=ticker)
                   )
               ]
           )
           num_points = client.count(
               collection_name=coll_name, count_filter=query_filter
           ).count
           log.info(f"{ticker:<4}\t{num_points}")
       ```
        NVDA	621  ;;;; AAPL	388 ;;;; MSFT	551 ;;;; AMZN	513 ;;;;
        GOOGL	670  ;;;; META	848 ;;;; TSLA	248 ;;;; WMT 	658 ;;;;
        JPM 	2673 ;;;; NFLX	517

    The client.scroll function has return type:
    tuple[list[types.Record], types.PointId | None]
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qdrant_path, coll_name = cfg.vector_db.qdrant_path, cfg.vector_db.collection_name
    client = QdrantClient(path=qdrant_path)
    # number of points in each of the company files
    for ticker in cfg.data.companies:
        query_filter = Filter(
            must=[
                FieldCondition(key="metadata.company", match=MatchValue(value=ticker))
            ]
        )
        points = client.scroll(
            collection_name=coll_name,
            scroll_filter=query_filter,
            with_payload=["metadata"],
            with_vectors=False,
            limit=3000,
        )[0]
        headers = {
            "Header 1": set(),
            "Header 2": set(),
            "Header 3": set(),
            "Header 4": set(),
        }
        header_keys = list(headers.keys())
        for record in points:
            metadata = record.payload["metadata"]
            for key in header_keys:
                if key in metadata:
                    headers[key].add(metadata[key])
        log.info(f"TICKER: {ticker}")
        log.info(headers)


def test_sec_api(cfg):
    r"""Check out the sec-api python library with provided free API calls.

    I have installed the `pip install sec-api` python library.
    It allows for 100 free api calls. It returns PDF files of the SEC filings.
    I have, as yet, downloaded the 10K pdf for Amazon.
    """
    from sec_api import PdfGeneratorApi

    pdf_api = PdfGeneratorApi(cfg.data.sec_api_key)
    for url in cfg.temporary.urls_10k:
        if "amzn" in url:
            continue
        name = url.split("/")[-1].split("-")[0]
        content = pdf_api.get_pdf(url)
        with open(f"data/sec_api/{name}.pdf", "wb") as f:
            f.write(content)
        log.info(f"Company {name} done.")


def compare_headers(cfg):
    r"""Compare the md files from sec-api pdf + mineru and edgartools markdown.

    I am comparing the amazon.md file headers derived from sec-api pdf download
    processed subsequently with minerU, and the markdown file obtained from
    edgartools.
    """
    edgar_fpath = r"data/edgar_tools_filings/AMZN-cleaned.md"
    mineru_fpath = r"data/sec_api/mineru/amzn/auto/amzn.md"

    log.info("EDGAR HEADERS")
    with open(edgar_fpath, "r") as f:
        for line in f:
            if line.startswith("#"):
                print(line, end="")

    log.info("MINERU HEADERS")
    with open(mineru_fpath, "r") as f:
        for line in f:
            if line.startswith("#"):
                print(line, end="")


def get_mineru_types():
    import json

    path = r"data/sec_api/mineru/amzn/auto/amzn_content_list.json"
    with open(path, "r") as f:
        content_list = json.load(f)
    content_types = set()
    for content in content_list:
        content_types.add(content["type"].strip())
    print(content_types)


def test_splitters(cfg):
    r"""Testing various splitters provided by Langchain.

    We are having to do the testing since langchain documentation is not very clear
    on what the various splitters do.

    I need a file with various text sentences in it, to do the testing.
    But might be better to derive that from the amzn.md file itself since it will
    be more representative.

    The number of characters in the modified (cleaned) amzn.md file are 244,904.

    SpacyTextSplitter:
    1. When I used default settings, the chunk sizes were really large. There were
       a total of 66 chunks created with defaults.
    2. chunk_size = 0 is not allowed (must be > 0). There were a total of 1205
       chunks created with chunk_size = 1, chunk_overlap = 0. This seems to create
       reasonably sized chunks, except for some chunks which are quite large.
    Below is using [2] above. I have chosen it as good enough for now.
    Most of the chunks are below 1000 characters, only 6 are larger. I will be
    trying to pass them to an LLM and split them further.

    I tried to use deepseek-r1:8b for the task and it seemed to work reasonably.
    """
    from hydra.utils import instantiate
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_text_splitters import SpacyTextSplitter
    from pydantic import BaseModel, Field

    class ListOfStrings(BaseModel):
        llm_chunks: list[str] = Field(
            description="list of chunked strings derived from the input string"
        )

    provider = instantiate(cfg.model)
    model = provider(model=cfg.model.name, temperature=cfg.model.temp)
    template = cfg.temporary.chunking_prompt
    prompt = ChatPromptTemplate.from_template(template)
    structured_llm = model.with_structured_output(ListOfStrings)
    # structured_llm = model.bind_tools([ListOfStrings])
    with open(cfg.temporary.amzn_cleaned_md_path, "r") as f:
        document_text = f.read()
    splitter = SpacyTextSplitter(chunk_size=1, chunk_overlap=0)
    chunks = splitter.split_text(text=document_text)
    for i in range(len(chunks)):
        if len(chunks[i]) > cfg.vector_db.chunk_size:
            formatted_prompt = prompt.format_messages(
                text_segment=chunk.replace("\n", " ")
            )
            response = structured_llm.invoke(formatted_prompt)
            chunks[i] = response.llm_chunks


def test_newlines(cfg):
    r"""Need to check if mineru md files always have newlines (1 or 2) after tables.

    Will check the following:
    1. Are all tables on their own separate lines? Or are there some lines which have
       tables but they either don't start with <table> or don't end with </table>?
    Ans: No.
    ```python
        table_start_tag, table_end_tag = "<table>", "</table>"
        num_mixed_lines = 0
        with open(cfg.temporary.amzn_md_path, "r") as f:
            for line in f:
                if table_start_tag in line or table_end_tag in line:
                    table_start = line.strip().startswith(table_start_tag)
                    table_end = line.strip().endswith(table_end_tag)
                    if (not table_start) or (not table_end):
                        num_mixed_lines += 1
        log.info(f"md file lines with table and other text {num_mixed_lines}")
    ```
    2.
    """

    table_start_tag, table_end_tag = "<table>", "</table>"
    num_mixed_lines = 0
    with open(cfg.temporary.amzn_md_path, "r") as f:
        for line in f:
            if table_start_tag in line or table_end_tag in line:
                table_start = line.strip().startswith(table_start_tag)
                table_end = line.strip().endswith(table_end_tag)
                if (not table_start) or (not table_end):
                    num_mixed_lines += 1
    log.info(f"Num lines in the md file with table and other text {num_mixed_lines}")


def get_table_description(cfg, table_image_path, company_name, table_context):
    r"""Returns a string description of a table given the image path and the context."""
    import base64
    from langchain_core.messages import HumanMessage

    provider = hydra.utils.instantiate(cfg.model)
    model = provider(model=cfg.model.name)
    # get image as base64
    with open(table_image_path, "rb") as file:
        base64_img = base64.b64encode(file.read()).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": cfg.prompts.table_description.format(
                    company_name=company_name, table_context=table_context
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
            },
        ]
    )
    table_description = model.invoke([message]).content.replace("\n", " ")
    return table_description


def process_tables(cfg):
    r"""Process the tables in the md files from mineru.

    Process tables:
    1. Take the markdown file and split it by single newlines - clean empty ones.
    2. Go through the content list file and for each table type in that json list
       find where it occurs in the md splits, then pass it and K prior non-table
       (since there might be more than one table one after another) elements to an
       LLM to get a summary - use the same prompt as the FinSage paper.
    3. Replace the summary in the place of the table and re-create the markdown
       file.

    Note: for now I am using just the hardcoded amazon paths. need to fix this to
          use more adaptable paths.
    """
    import json

    # read the markdown file into a list of strings by splitting at double newline
    with open(cfg.temporary.amzn_md_path, "r") as f:
        split_text = f.read().split("\n\n")
    # clean each split to remove whitespace from the ends - precautionary
    for i, split in enumerate(split_text):
        split_text[i] = split.strip()
    # obtain the json contents list
    with open(cfg.temporary.amzn_content_list_path, "r") as f:
        content_list = json.load(f)
    # verify that the number of tables in the markdown file are the same as the
    # number of tables in the content list
    table_start = "<table>"
    num_tables_md = sum([1 if table_start in split else 0 for split in split_text])
    num_table_imgs = sum([1 if it["type"] == "table" else 0 for it in content_list])
    assert num_tables_md == num_table_images
    # verify that the tables of the content list all start with <table> and end
    # with the </table> tag.
    table_end = "</table>"
    for content in content_list:
        if content["type"] == "table":
            table = content["table_body"].strip()
            assert table.startswith(table_start) and table.endswith(table_end)
            content["table_body"] = table  # strip it here itself
    # verify that every single table from the content list is present in the md file.
    for content in content_list:
        if not content["type"] == "table":
            continue
        table = content["table_body"]
        assert table in split_text
    # for content in content_list:
    #     if not content["type"] == "table":
    #         continue
    #     table = content["table_body"]
    #     # specifically use index since we want to fail if it does not occur
    #     text_idx = split_text.index(table)


def build_pre_processing_pipeline(cfg):
    r"""This will try and do all the preprocessing after mineru to fill vector db.

    Process tables:
    1. Take the markdown file and split it by single newlines - clean empty ones.
    2. Go through the content list file and for each table type in that json list
       find where it occurs in the md splits, then pass it and K prior non-table
       (since there might be more than one table one after another) elements to an
       LLM to get a summary - use the same prompt as the FinSage paper.
    3. Replace the summary in the place of the table and re-create the markdown
       file.

    Create chunks:
    1. Load the un-tabled markdown file and split it using markdown header splitter
       giving documents.
    2. There might be some sections that have no content in them - need to see what
       to do about them.
    3. For each document, use an LLM to generate a summary of that section.
    4. Use the spacy splitter on each document to generate sentence chunks. If any
       chunk is larger than the configured chunk size, use an LLM to break it up
       into manageable sentences.
    5. For each summary add the year, company, header(s), type as metadata and add
       them to the vector db.
    5. For each chunk add the year, company, header(s), summary, type as metadata
       and add them to the vector db.
    """
    pass


if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path=".."):
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    sub_query = "What are the risk factors for apple?"
    test_image_and_text(cfg)
