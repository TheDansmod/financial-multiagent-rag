r"""Split markdown files, get segment summaries, and populate vector db with the chunks.

1. First, split the markdown files by the markdown header to obtain segment documents.
   Obtain summaries for each segment using an LLM.
2. Then, split the segments sentence by sentence - using an LLM where needed.
3. Then, decontexualise each sentence by passing the sentence and some context (previous
   sentences) to an LLM. Each decontextualised sentence is now a chunk, as are all of
   the summaries.
4. Embed both, the sentences and the summaries into the vector database. Ensure that the
   summary is part of the metadata of each sentence.

Rough (TODO - delete later):
    1. Generate section summaries function - split md by header, get per section summary
       and store the summaries in a json file
    2. Decontexualise function - split each section by sentences, decontextualise each
       sentence and store the result in a json file
    3. Populate vector db function - embed both the decontextualised sentences and the
       summaries and populate a vector db with them
"""
import logging
from src.utils.utils import get_newline_split_markdown
from langchain_text_splitters import MarkdownHeaderTextSplitter

log = logging.getLogger(__name__)

# TODO: delete below 2 function later
# i am doing below since my idea is that we can obtain sections by splitting on newlines without using the markdown header text splitter
def get_text(path):
    with open(path, 'r') as f:
        return f.read()

def check_headers(cfg):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    valid_headers = ["Header 1", "Header 2", "Header 3", "Header 4"]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    for ticker in cfg.data.companies:
        md_file_path = cfg.pre_proc.processed_md_file_path.format(ticker=ticker.lower())
        md_text = get_text(md_file_path)
        split_text = get_newline_split_markdown(md_file_path)
        newline_headers = set([split for split in split_text if split.startswith('#')])
        splitter_headers = set()
        docs = md_splitter.split_text(text=md_text)
        for doc in docs:
            for k, v in doc.metadata.items():
                if k in valid_headers:
                    splitter_headers.add(v)
        print(f"{ticker}: {len(newline_headers) - len(splitter_headers)}")
        stripped_newline_headers = set([hdr.replace('# ', '') for hdr in newline_headers])
        for elem in (stripped_newline_headers - splitter_headers):
            print(f"\tNewline: {elem}")
        for elem in (splitter_headers - stripped_newline_headers):
            print(f"\tSplitter: {elem}")
        print("\n\n\n")

if __name__ == "__main__":
    import hydra
    import time
    from dotenv import load_dotenv
    from omegaconf import DictConfig
    
    t0 = time.perf_counter()
    load_dotenv()
    t1 = time.perf_counter()
    with hydra.initialize(version_base=None, config_path="../../config/"):
        t2 = time.perf_counter()
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
        t3 = time.perf_counter()
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    t4 = time.perf_counter()
    check_headers(cfg)
    t5 = time.perf_counter()

    print(f"dotenv loading: {t1 - t0:.6f}s")
    print(f"hydra initialise: {t2 - t1:.6f}s")
    print(f"hydra compose: {t3 - t2:.6f}s")
    print(f"hydra log conf: {t4 - t3:.6f}s")
    print(f"check headers: {t5 - t4:.6f}s")
    print(f"total time: {t5 - t0:.6f}s")
