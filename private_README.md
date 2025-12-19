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

## Steps taken - Advanced Project
### data download and understanding
1. Installed `pip install sec-edgar-downloader` to obtain the SEC 10K filing data - in the folder `data/edgar_tools_filings/`, the function used: `utils.get_markdown_sec_filings`
2. Wrote the downloader (in `utils.py` file) and downloaded data for 10 companies. See the structure of the document below. Item 1, 1A, 7 are the most important parts.
3. See possible chunking strategies below - I will be going with Parent Document approach. After converting the HTML to markdown since Llama 3.1 and similar LLMs are heavily trained on markdown
4. Installed `pip install beautifulsoup4 markdownify lxml` to parse the HTML data into markdown with the superior lxml library backend for bs4
5. There were some div tags present in the markdown, I wrote some code to clean them up. Function used: `utils.clean_edgar_markdown`


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

## Chunking strategies
1. Recursive chunking - baseline - paragraph, sentence, space - good for general text sections
2. Semantic chunking - break chunk where similarity falls below threshold - good for dense sections like business overview where topics shift rapidly
3. Structure-Aware chunking - beautiful soup to chunk by headers - needs secondary chunking inside sections (Parent-Document)
