"""Build the mulit-agent RAG system using LangGraph.

We will be building the following multi-agent RAG system:
1. Supervisor (Agent A): classifies the query into simple, comparison, and complex;
   and identifies the relavant companies. Sends simple queries to agent C, and
   others to Agent B.
2. Planner (Agent B): Divides each comparison or complex query into a list of
   sub-queries, which when answered, together answer the original user query. Each
   sub-query is independently sent to the retriever.
3. Retriever (Agent C): For each sub-query, it gets the top K matching chunks from
   the vector db and aggregates and passes them to the LLM to generate an answer
   for this subquery. Then we move to agent D.
4. Grader (Agent D): For each sub-query and corresponding answer along with the
   aggregated context, it checks if the sub-query has been answered correctly and
   there are no facts used that are not present in the aggregated context. If yes,
   we move to Agent E. Else, it gives a short comment on what to correct and we go
   back to Agent C. The D-C loop happens at most M times (M could be a small value
   like 3)
5. Consolidator (Agent E): Aggregate the answers to all the subqueries so as to
   answer the original user query.
"""

import logging
from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from hydra.utils import instantiate
from langchain_core.prompts import ChatPromptTemplate

log = logging.getLogger(__name__)


class QueryType(TypedDict):
    """Classify the original user query by company and difficulty.

    TODO: add more description
    """

    query_complexity: Literal["simple", "comparison", "complex"]
    query_companies: list[
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
            "any",
        ]
    ]


class SubQueries(TypedDict):
    """This class is created to act as a template for the LLM response.

    TODO: add more description
    """

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


class SubQuery(TypedDict):
    """The Agent State related to each sub-query.

    TODO: add more description
    """

    # main_sub_query: Literal[MainSubQuery]
    contexts: list[str] | None
    answer: str | None
    times_graded: int = 0


class AgentState(TypedDict):
    """The state object passed around by the multi-agent setup.

    TODO: add more description
    """

    user_query: str
    query_type: QueryType | None
    sub_queries: list[SubQuery] | None
    answer: str | None


class FinancialAnalystAgent:
    """The actual multi-agent RAG system.

    TODO: add more description.
    """

    def __init__(self, cfg):
        model_provider = instantiate(cfg.models.analyst)
        log.info(model_provider)
        self.model = model_provider(
            model=cfg.models.analyst.name, temperature=cfg.models.analyst.temp
        )
        self.cfg = cfg

    # def supervisor(self, state: AgentState) -> Command[Literal["planner", "retriever"]]:
    def supervisor(self, state: AgentState) -> Command[Literal[END]]:
        template = self.cfg.prompts.supervisor_node
        prompt = ChatPromptTemplate.from_template(template)
        formatted_prompt = prompt.format_messages(user_query=state["user_query"])
        structured_llm = self.model.with_structured_output(QueryType)
        classification = structured_llm.invoke(formatted_prompt)
        # if classification["query_complexity"] == "simple":
        #     next_node = "retriever"
        # else:
        #     next_node = "planner"
        # return Command(update={"query_type": classification}, goto=next_node)
        return Command(update={"query_type": classification}, goto=END)

    # def planner(self, state: AgentState) -> AgentState:
    #     template = self.cfg.agent_configs.planner_node.system_prompt_template
    #     prompt = ChatPromptTemplate.from_template(template)
    #     formatted_prompt = prompt.format_messages(user_query=state["user_query"])
    #     structured_llm = self.model.with_structured_output(MainSubQuery)
    #     main_sub_query = structured_llm.invoke(formatted_prompt)

    def invoke(self, initial_state, config):
        builder = StateGraph(AgentState)

        # add nodes
        builder.add_node("supervisor", self.supervisor)

        # add edges
        builder.add_edge(START, "supervisor")
        builder.add_edge("supervisor", END)
        app = builder.compile()
        result = app.invoke(initial_state, config)
        return result

if __name__ == "__main__":
    import hydra
    from dotenv import load_dotenv
    from omegaconf import DictConfig

    # boiler-plate
    load_dotenv()
    with hydra.initialize(version_base=None, config_path="../../config/"):
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    
    # file code
    agent = FinancialAnalystAgent(cfg)
    initial_state = {"user_query": "What is the percentage increase in net sales in the US between 2024 and 2025 for Apple?"}
    config = {"configurable": {"thread_id": "customer_123"}}
    log.info(f"RESULT:\n{agent.invoke(initial_state, config)}")
