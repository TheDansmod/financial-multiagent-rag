## Steps taken
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
17. Also installed langchain-classic to use some of its functions
