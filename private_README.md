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
