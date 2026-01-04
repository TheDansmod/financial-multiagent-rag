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
    Ans: All tables are on their own separate lines.
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
       file - this is done in a different function - populate_markdown

    Note: The "table" type in content list has relevant keys: table_body (actual), 
          table_footnote (list[str]), table_caption (list[str]), img_path (str).
    Note: If you ignore table types in content list that don't have a table_body
          key, there is a bijection between the markdown file tables and content
          list tables.
    Note: This function is only obtaining the responses from the LLM, not updating
          the markdown files.
    TODO: add the variable that controls how many prior splits of the markdown file
          you will pass as context to the LLM for obtaining the table description
          to the config file. Here, the value is 3.
    TODO: remove blank splits from split_text
    TODO: refactor the function to be more readable and modular
    """
    import json
    from pathlib import Path

    table_start = "<table>"
    table_end = "</table>"
    for ticker in cfg.data.companies:
        md_path = cfg.data.md_file_path.format(ticker=ticker.lower())
        content_list_path = cfg.data.content_list_file_path.format(
            ticker=ticker.lower()
        )
        images_folder = cfg.data.images_folder_path.format(ticker=ticker.lower())
        # read the markdown file into a list of strings by splitting at double newline
        with open(md_path, "r") as f:
            split_text = f.read().split("\n\n")
        # clean each split to remove whitespace from the ends - precautionary
        for i, split in enumerate(split_text):
            split_text[i] = split.strip()
        # verify that all tables are separate elements in the markdown file
        for split in split_text:
            if (table_start not in split) or (table_end not in split):
                continue
            assert split.startswith(table_start)
            assert split.endswith(table_end)
        # obtain the json contents list
        with open(content_list_path, "r") as f:
            content_list = json.load(f)
        # verify that the number of tables in the markdown file are the same as the
        # number of tables in the content list, and they all have an image
        # also that each table in the content list is present somewhere in the md file
        # overall this means we have a bijection between the tables in the markdown
        # file and the tables in the content list file.
        num_tables_md = sum([1 if table_start in split else 0 for split in split_text])
        num_content_tables = 0
        for content in content_list:
            if not (content["type"] == "table" and "table_body" in content):
                continue
            num_content_tables += 1
            assert content["table_body"] in split_text
            assert content["img_path"].strip()  # should not be blank
        # get the description for each table in the markdown file
        for idx, split in enumerate(split_text):
            if table_start not in split:
                continue
            # get this table's data and image path from the content list file
            for content in content_list:
                if "table_body" in content and content["table_body"] == split:
                    caption = (" ".join(content["table_caption"])).strip()
                    footnote = (" ".join(content["table_footnote"])).strip()
                    image_path = Path(f"{images_folder}{content['img_path'].split('/')[1]}")
                    break
            # get the previous 3 non-table splits from the markdown file
            # don't add any previous splits if you encounter a heading
            splits_needed, splits_obtained = 3, 0
            table_context_md = []
            for i in range(idx-1, -1, -1):
                # we only want non-tabular splits
                if table_start in split_text[i]:
                    continue
                # stop if we have the number of needed splits
                if splits_obtained >= splits_needed:
                    break
                # add the header and no more if there is a header
                if split_text[i].startswith('#'):
                    table_context_md.append(split_text[i])
                    break
                table_context_md.append(split_text[i])
                splits_obtained += 1
            # splits were added in reverse order, putting them right
            md_context = (" ".join(table_context_md[::-1])).strip()
            table_context = ""
            if caption:
                table_context += f"Caption: {caption}\n"
            if footnote:
                table_context += f"Footnote: {footnote}\n"
            if md_context:
                table_context += f"Text before the table in the filing: {md_context}"
            table_description = get_table_description(cfg, image_path, ticker, table_context)
            # put all this into a file so that if anything fails, I don't have to repeat the LLM call for those that are already done
            table_json = {"Image": str(image_path), "Ticker": ticker, "Table Context": table_context, "Split Index": idx, "Split": split, "Table Description": table_description}
            # json data is a list
            with open(cfg.data.table_descriptions_path, 'r') as file:
                json_data = json.load(file)
            json_data.append(table_json)
            with open(cfg.data.table_descriptions_path, 'w') as file:
                json.dump(json_data, file)

def populate_markdown(cfg):
    r"""Replace the tables in the markdown files with the downloaded descriptions.

    In the other function, I found the tables in the markdown files, obtained the
    context, sent them to the LLM and populated a json file with the responses.
    In this function, we will read the json file and replace the content in the
    markdown files.
    """
    import json

    with open(cfg.data.table_descriptions_path, "r") as file:
        json_data = json.load(file)
    # for each company, load the md file as splits, for each table in the splits
    # replace it with the correct description from the json file by iterating
    # through the json file
    for ticker in cfg.data.companies:
        # load md file and generate splits
        md_path = cfg.data.md_file_path.format(ticker=ticker.lower())
        proc_md_path = cfg.data.processed_md_file_path.format(ticker=ticker.lower())
        with open(md_path, "r") as f:
            split_text = f.read().split("\n\n")
        # clean each split to remove whitespace from the ends - precautionary
        for i, split in enumerate(split_text):
            split_text[i] = split.strip()
        for table_descr in json_data:
            if table_descr["Ticker"] == ticker:
                split_text[table_descr["Split Index"]] = table_descr["Table Description"]
        proc_md_file = "\n\n".join(split_text)
        with open(proc_md_path, "w") as file:
            file.write(proc_md_file)

def create_header_split_file(cfg):
    r"""Splits the markdown files into sections based on headers and gets their
    summaries from an LLM and adds them to the header splits json file.

    1. The header splits file must exist (but can be empty).
    2. We initially ensure that the header splits file contains valid json.
    3. We iterate through all the markdown files, and generate header splits for each
       of them.
    4. For each header file, for each split in it, we read the json data (again since
       it might have changed), and check if the split has already been summarised. If
       yes, then we continue on to the next split. If not, then we summarise the
       split by llm invocation, add a dict for this split to the json data, and write
       the json data back to file.
    5. We read and write the json data for each split since we want to persist every
       response of the LLM since that is expensive.

    Since there are only 2210 total sections across the 10 markdown files, I feel it
    would be ok to use the gemma-3-27b-it model for this task.

    TODOs:
    1. Add code to ensure that header splits file need not even exist when this
       code is run.
    """
    import json
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
    from uuid import uuid4

    # we ensure file content is valid json
    with open(cfg.data.header_splits_file_path, "r") as file:
        file_data = file.read().strip()
    if not file_data:
        json_data = []
    else:
        json_data = json.loads(file_data)
    with open(cfg.data.header_splits_file_path, "w") as file:
        json.dump(json_data, file)

    provider = hydra.utils.instantiate(cfg.model)
    llm = provider(model=cfg.model.name)
    prompt = ChatPromptTemplate.from_template(cfg.prompts.header_section_summary)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    valid_headers = ["Header 1", "Header 2", "Header 3", "Header 4"]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    counter = len(json_data)
    for ticker in cfg.data.companies:
        with open(cfg.data.processed_md_file_path.format(ticker=ticker.lower()), "r") as file:
            text = file.read()
        header_splits = md_splitter.split_text(text)
        for split in header_splits:
            # check if split is present in json file and has summary
            with open(cfg.data.header_splits_file_path, "r") as file:
                json_data = json.load(file)
            split_summarised = False
            for elem in json_data:
                # each elem is a dictionary with keys ticker, page_content, id, summary (if summarised), metadata
                if elem["ticker"] == ticker and elem["page_content"] == split.page_content and "summary" in elem and elem["summary"].strip() and elem["metadata"] == split.metadata:
                    split_summarised = True
                    break
            if split_summarised:
                continue
            section_headings = " ".join([v for k, v in split.metadata.items() if k in valid_headers])
            formatted_prompt = prompt.format_messages(ticker=ticker, section_headings=section_headings, text_section=split.page_content)
            split_summary = llm.invoke(formatted_prompt).content
            split_json = {"ticker": ticker, "page_content": split.page_content, "id": str(uuid4()), "metadata": split.metadata, "summary": split_summary}
            json_data.append(split_json)
            with open(cfg.data.header_splits_file_path, "w") as file:
                json.dump(json_data, file)
            counter += 1
            log.info(f"Done {counter} of 2210")


def build_pre_processing_pipeline(cfg):
    r"""This will try and do all the preprocessing after mineru to fill vector db.

    Process tables: - process_tables, populate_markdown
    1. Take the markdown file and split it by single newlines - clean empty ones.
    2. Go through the content list file and for each table type in that json list
       find where it occurs in the md splits, then pass it and K prior non-table
       (since there might be more than one table one after another) elements to an
       LLM to get a summary - use the same prompt as the FinSage paper.
    3. Replace the summary in the place of the table and re-create the markdown
       file.
    Note: The above is done (01 Jan 2026 14:00 PM) and files are stored in the
          cfg.data.processed_md_file_path variable

    Splitting each file into chunks and getting the summary of each chunks is done
    by a different function - create_header_split_file

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

    Need to do co-reference resolution, which can't be done with the json file since
    the elements in the json file are not necessarily in order, and for co-reference
    resolution we need the previous 10 non-tabular chunks. I think I will have to do
    markdown header splitting again.

    Since there are likely quite a few sentences overall, I will likely be using the
    deepseek model for doing decontexualization. I am also using deepseek model for
    dividing larger chunks into smaller ones since I need a structured LLM for that.

    Results:
    1. The header splitter produces 2210 total sections, with max len 49,832 and min
       len 4. Only 16 larger than 20k, and 91 larger than 10k.
    2. The number of unique elements of the header splitter (including the ticker,
       page_content, and metadata) are 2195.
    3. The total number of calls to llms for splitting large chunks was 64 and the
       total number of calls to llms for de-contextualization was 27,952.
    4. I would like to test how long it will take on my local PC. So will do random
       calls for 0.001 fraction of the total.
    5. I also want to figure out what the number of tokens are for all my prompts
    6. The largest token size is 1380 tokens, no tokens larger than 4096.
    7. Based on 17 random invocations, the average input tokens is 661.65, the avg
       output tokens is 570.65, the speed with deepseek-r1:8b is 43.68 tokens/sec,
       and the estimated time for 30k calls is 108.88 hours. Not viable.

    TODOs:
    1. Currently I have pre-populated the empty header splits file with a list so
       that json loading works. Insert code for handling empty or non-existent file.
    2. Set the number of context chunks as a parameter in the config file
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document
    from langchain_text_splitters import (
        SpacyTextSplitter,
        MarkdownHeaderTextSplitter,
    )
    from pydantic import BaseModel, Field
    from uuid import uuid4
    import json
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    from frozendict import frozendict
    import numpy as np
    import time

    class ListOfStrings(BaseModel):
        llm_chunks: list[str] = Field(
            description="list of chunked strings derived from the input string"
        )

    def order_header_splits_json(json_data):
        """The cfg.data.header_splits_file_path file might contain elements that are
        not in order, but the decontexualization needs them in order, so this
        function adds order to them.
        """
        log.info(f"number of elements in header splits file: {len(json_data)}")
        splits_seq = []
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        for ticker in cfg.data.companies:
            with open(cfg.data.processed_md_file_path.format(ticker=ticker.lower()), "r") as file:
                text = file.read()
            # split_text returns list of docs, but we only want the text 
            # - added ticker to hopefully make it unique
            header_splits = [(ticker, doc.page_content, frozendict(doc.metadata)) for doc in md_splitter.split_text(text=text)]
            splits_seq.extend(header_splits)
        # since python 3.7 dictionaries guarantee preservation of insertion order
        unique_splits_seq = list(dict.fromkeys(splits_seq))
        log.info(f"number of elements in splits seq: {len(splits_seq)}")
        log.info(f"number of elements that are unique in splits seq: {len(set(splits_seq))}")
        new_json_data = [None] * len(json_data)
        for elem in json_data:
            # use index in place of find since we want it to error out if the split
            # is absent
            idx = unique_splits_seq.index((elem["ticker"], elem["page_content"], frozendict(elem["metadata"])))
            new_json_data[idx] = elem
        return new_json_data

    def get_decontexualization_context(main_doc):
        r"""In this function we will iterate in reverse through the documents list
        and add at most num_context_chunks elements to context_list. 

        I was planning on skipping tables, but that is not really possible since how
        will I even figure out what is a table since all I have are sentences. The
        summary might actually contain a table, but that does not necessarily mean
        that the current sentence is part of a table.

        I was also planning on terminating if I encounter any headers, and that is
        I think actually possible - I can keep checking the header to see if it has
        changed. Actually since we don't know the header title, might be better to
        use summary and ticker.
        """
        context_list = []
        for doc in documents[:-(num_context_chunks + 1):-1]:
            if not (doc.metadata["section_summary"] == main_doc.metadata["section_summary"] or doc.metadata["ticker"] == main_doc.metadata["ticker"]):
                break
            context_list.append(doc.page_content)
        return " ".join(context_list)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    spacy_splitter = SpacyTextSplitter(chunk_size=1, chunk_overlap=0)
    decon_template = cfg.prompts.de_contexualization
    decon_prompt = ChatPromptTemplate.from_template(decon_template)
    chunking_template = cfg.prompts.chunking
    chunking_prompt = ChatPromptTemplate.from_template(chunking_template)
    provider = hydra.utils.instantiate(cfg.model)
    llm = provider(model=cfg.model.name)
    structured_llm = llm.with_structured_output(ListOfStrings)
    log.info("obtained prompts and llms")

    num_context_chunks = 10
    with open(cfg.data.header_splits_file_path, "r") as file:
        json_data = order_header_splits_json(json.load(file))
    log.info("obtained new json data with right order")
    # first we encode and add the chunks, then we'll do the metadata
    documents = []
    num_llm_calls = 0
    for elem in json_data:
        chunks = spacy_splitter.split_text(text=elem["page_content"])
        for chunk in chunks:
            if len(chunk) > cfg.vector_db.chunk_size:
                chunking_formatted_prompt = chunking_prompt.format_messages(text_segment=chunk.replace("\n", " "), ticker=elem["ticker"])
                split_chunks = strctured_llm.invoke(chunking_formatted_prompt).llm_chunks
                for split_chunk in split_chunks:
                    metadata = elem["metadata"] | {"type": "content", "section_summary": elem["summary"], "ticker": elem["ticker"]}
                    doc = Document(page_content=split_chunk, metadata=metadata, id=str(uuid4()))
                    decon_formatted_prompt = decon_prompt.format_messages(ticker=elem["ticker"], main_chunk=doc.page_content, chunk_context=get_decontexualization_context(doc))
                    doc.page_content = llm.invoke(decon_formatted_prompt).content
                    documents.append(doc)
            else:
                metadata = elem["metadata"] | {"type": "content", "section_summary": elem["summary"], "ticker": elem["ticker"]}
                doc = Document(page_content=chunk, metadata=metadata, id=elem["id"])
                decon_formatted_prompt = decon_prompt.format_messages(ticker=elem["ticker"], main_chunk=doc.page_content, chunk_context=get_decontexualization_context(doc))
                doc.page_content = llm.invoke(decon_formatted_prompt).content
                documents.append(doc)
    # now summaries at the end - no decontexualization for them since they are already expected to be decontexualised since they are summaries
    for elem in json_data:
        summary_metadata = elem["metadata"] | {"type": "summary", "ticker": elem["ticker"]}
        summary_doc = Document(page_content=elem["summary"], metadata=summary_metadata, id=str(uuid4()))
        documents.append(summary_doc)

    # add these documents to the vector db
    embedding_model = cfg.vector_db.embedding_model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    client = QdrantClient(path=cfg.vector_db.qdrant_path)
    
    collection_name = cfg.vector_db.collection_name
    if not client.collection_exists(collection_name):
        log.info(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=client.get_embedding_size(embedding_model), distance=Distance.COSINE),
        )
    else:
        log.info(f"Collection '{collection_name}' already exists. Appending documents...")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    vector_store.add_documents(documents=docs)


if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path=".."):
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    sub_query = "What are the risk factors for apple?"
    # create_header_split_file(cfg)
    build_pre_processing_pipeline(cfg)
