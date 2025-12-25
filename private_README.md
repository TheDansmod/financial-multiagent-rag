## Steps taken - basic project
1. Installed ollama on arch linux: `paru -S ollama-cuda`
2. Started up ollama: `ollama serve`
3. On a different terminal obtained llama 3.1: `ollama pull llama3.1`
4. After download, ran the model: `ollama run llama3.1`
5. Created python environment with 3.13 in folder `agentic_rag_3-13`
6. Activated the environment: `source linux_environments/agentic_rag_3-13/bin/activate`
7. Installed qdrant: `pip install qdrant-client`
8. Installed langchain tooling: `pip install langchain langchain-community langchain-qdrant langchain-huggingface langchain-ollama`
9. Installed the ChatGPT-like UI provider: `pip install chainlit`
10. Installed the pdf parser and the huggingface embedding creator: `pip install chainlit sentence-transformers`
11. Created the `data` folder and added the `my_knowledge.txt` file to it
12. Also installed `langgraph-cli[inmem]` since that was required for the langgraph essentials course
13. Also installed `jupterlab` since that was required for the langgraph essentials course - jupyter lab can be started in the browser by just running `jupyter lab` in the terminal in the correct folder
14. Created the qdrant logic in `ingest.py` and the app logic in `app.py`
15. Also installed `fastembed` for qdrant - it gave error saying it needs it
16. I have run `python ingest.py` which has generated the persistent qdrant vector store
17. I have run application: `chainlit run app.py -w` where `-w` helps perform automatic reload in case of code update
18. I have also installed `pip install ruff hydra-core hydra_colorlog` as utilities to help with linting and formatting (ruff) and config setup (hydra) and colourful logging with hydra (colorlog)

## Steps taken - Advanced Project
### data download and understanding
1. Installed `pip install sec-edgar-downloader edgartools` to obtain the SEC 10K filing data - in the folder `data/edgar_tools_filings/`, the function used: `utils.get_markdown_sec_filings` (I used edgar tools since that directly gives the markdown rather than the sec-edgar-downloader which gives a txt result).
2. Wrote the downloader (in `utils.py` file) and downloaded data for 10 companies. See the structure of the document below. Item 1, 1A, 7 are the most important parts.
3. See possible chunking strategies below - I will be going with Parent Document approach. After converting the HTML to markdown since Llama 3.1 and similar LLMs are heavily trained on markdown
4. Installed `pip install beautifulsoup4 markdownify lxml` to parse the HTML data into markdown with the superior lxml library backend for bs4
5. There were some div tags present in the markdown, I wrote some code to clean them up. Function used: `utils.clean_edgar_markdown`
6. Added a function to split the markdown with metadata and add the splits into a vector db after embedding. Function used: `utils.generate_qdrant_db`. I have also incorporated hydra configuration manager into this with the compose API since I might actually invoke utils from a main file.
7. Added a function `utils.test_llama_json_handling` to check how well the model handles json queries - it handles them fine
8. There is a good multi-pronged chunking strategy highlighted in the appendix, but for now I will focus on getting the current version working with a RAG agent.
9. For the multi-agent RAG, I have provided a basic framework in Approach 01, which I will be following for now, to obtain a proof-of-concept outcome.
10. I have created a project called `multi-agent-rag` on google ai studio and for that project I have created a key named `danish-multi-agent-rag` which I will be using for this project. I have also install `pip install langchain-google-genai` to try and use google's flash model.
11. For now, I am trying to use gemma-3-27b model rather than the gemini-3-flash-preview model
12. I have also installed `pip install sec-api` and created a free API with them since they provide 100 free API calls and SEC-10K filings in PDF format. I tried to download amazon SEC-10K filing using their API which seems to have gone fine. I am also installing `pip install mineru[core]` which can process pdf documents and is used by one of the papers I looked at.


## Appendix
### The SEC 10K document:
1. Item 1 (Business): Describes the main products/services, subsidiaries, markets, and competition.
2. Item 1A (Risk Factors): Lists risks that could hurt the stock (regulation, competition, supply chain, etc.)
3. Item 1B (Unresolved Staff Comments): generally says none
4. Item 1C (Cybersecurity): Describes risk management and governance regarding cybersecurity threats.
5. Item 2: (Properties): Lists physical assets (factories, HQs, data centers, mines). Tech giants (Google/Meta) list data centers here; Retailers (Walmart) list store counts.
6. Item 3: (Legal Proceedings): Discloses significant pending lawsuits or government investigations. Only material lawsuits are listed.
7. Item 4: (Mine Safety Disclosures): mining operations related. none of the tech companies will generally have it
8. Item 5: (Market for Common Equity): Stock price history, dividends paid, and stock repurchases (buybacks). Look here to see how much stock the company bought back.
9. Item 7 (MD&A - Management’s Discussion and Analysis): The CEO explaining why numbers went up or down. This is the narrative explanation of the financial results.
10. Item 8 (Financial Statements): Raw tables. The audited Balance Sheet, Income Statement, and Cash Flow Statement.
11. Item 9 (Changes in Accountants): generally none
12. Item 9A (Controls and Procedures): Certification that internal financial controls (accounting software/processes) work.
13. Item 9B/9C (Other Information)
14. Part III: Items 10-14 (Governance and Pay): Generally filed later

### Chunking strategies
1. Recursive chunking - baseline - paragraph, sentence, space - good for general text sections
2. Semantic chunking - break chunk where similarity falls below threshold - good for dense sections like business overview where topics shift rapidly
3. Structure-Aware chunking - beautiful soup to chunk by headers - needs secondary chunking inside sections (Parent-Document)
4. Multi-pronged semantic strategy:
    1. We are already doing the parent-child hierarchical chunking, we might explore if it would be beneficial to prepend the metadata about headings and company to the child rather than having to handle that in the RAG agent.
    2. We can use the llama model to split the more complex sentences into atomic propositions: 
    The sentence: "Revenue increased 10% due to higher sales volume, partially offset by a 5% increase in COGS." 
    can be split into the following:
        1. "Revenue increased 10% due to higher sales volume."
        2. "Revenue increase was partially offset by a 5% increase in COGS."
    note that this is different from just splitting the sentence on the comma, and makes each split sentence coherent.
    We can then do embedding for these split sentences
    3. Tables should be isolated, replaced with their summaries, and when matched, the full table should be passed into the LLM context

### Agentic Setups:
#### Approach 01:
1. Supervisor (Agent A): classifies the query into simple, comparison, and complex; and identifies the relavant companies. Sends simple queries to agent C, and others to Agent B.
2. Planner (Agent B): Divides each comparison or complex query into a list of sub-queries, which when answered, together answer the original user query. Each sub-query is independently sent to the retriever.
3. Retriever (Agent C): For each sub-query, it gets the top K matching chunks from the vector db and aggregates and passes them to the LLM to generate an answer for this subquery. Then we move to agent D.
4. Grader (Agent D): For each sub-query and corresponding answer along with the aggregated context, it checks if the sub-query has been answered correctly and there are no facts used that are not present in the aggregated context. If yes, we move to Agent E. Else, it gives a short comment on what to correct and we go back to Agent C. The D-C loop happens at most M times (M could be a small value like 3)
5. Consolidator (Agent E): Aggregate the answers to all the subqueries so as to answer the original user query.

### Current Questions
1. What sorts of headings are present across all the markdown files?
2. How has the situation where there are multiple headings at the same level been handled?
3. How has the situation where there are headings without content been handled?
4. Deepseek R1 8B is also a viable model but it thinks more. Can try using that. I have downloaded it. (deepseek-r1:8b)


