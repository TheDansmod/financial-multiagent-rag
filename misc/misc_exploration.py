"""Exploring various things in the RAG - they are tested out here first.

In this file we will be trying out various things to see how they work, or we will be
exploring the datasets and techniques etc.
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


if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path=".."):
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    fact_query = "Who is the auditor for Apple in 2023?"
    comparison_query = "Compare the risk factors for apple and microsoft."
    complex_query = (
        "What is the primary risk factor listed by the company"
        "with the highest revenue in the tech sector?"
    )
    analyse_headers(cfg)
