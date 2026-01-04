r"""Fetch the data and parse the obtained PDF files.

1. Fetch Data: Obtain the pdf files of the SEC10K filings from SEC API using the free
   API limit.
2. Parse PDFs: Parse the fetched pdf files into markdown and json so the are more
   easily ingested and processed by LLMs. We use the minerU library to do this.
"""

import logging
import subprocess

from sec_api import PdfGeneratorApi

log = logging.getLogger(__name__)


def fetch_data(cfg):
    r"""Fetch the PDF files for the SEC10K filings for the listed companies.

    I am using the free API provided by sec-api where they allow upto 100 free
    API calls. Since I only need the 10K documents for 10 companies, this works.
    """
    pdf_api = PdfGeneratorApi(cfg.data.sec_api_key)
    for ticker, url in zip(cfg.data.companies, cfg.data.urls_10k):
        name = ticker.lower()
        content = pdf_api.get_pdf(url)
        with open(cfg.data.pdf_file_path.format(ticker=name), "wb") as f:
            f.write(content)
        log.info(f"Company {name} done.")


def parse_pdf_files(cfg):
    r"""Parse SEC10K pdf files into markdown, contents list, images, using minerU.

    The backend can be specified to be either `vlm-transformers` or `pipeline`. I
    personally prefer the vlm backend since it is said to be more accurate, but it
    does take quite a while to run.
    Also, there is a python API for mineru, but for now I have not checked out how
    to use it, so I am instead using the commandline method of invocation, just from
    python.

    After being processed by minerU, the folder for each company (minerU creates a new
    folder for every file being processed) contains the following relevant files:
    (1) a markdown file which is the main content of the pdf input,
    (2) an images folder which contains images of tables or figures from the pdf input
    (3) a content_list.json file which is a list of dictionaries, each of which can be
        of some type (like image, text, table, discard, etc) and has the content for
        that type. For a table, it contains the text of the table and the path to the
        image of that table.
    """
    for ticker in cfg.data.companies:
        pdf_path = cfg.data.pdf_file_path.format(ticker=ticker.lower())
        output_folder = cfg.pre_proc.mineru_folder_path
        backend = cfg.pre_proc.mineru_backend
        command = ["mineru", "-p", pdf_path, "-o", output_folder, "-b", backend]
        try:
            subprocess.run(command, check=True)
            log.info(f"MinerU parsed pdf file for {ticker}.")
        except subprocess.CalledProcessError as e:
            log.error(f"MinerU failed to parse pdf file for {ticker}. Continuing.")
            log.error(f"Error:\n{e.stderr}")


if __name__ == "__main__":
    import hydra
    from dotenv import load_dotenv
    from omegaconf import DictConfig

    load_dotenv()
    with hydra.initialize(version_base=None, config_path="../../config/"):
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    fetch_data(cfg)
    parse_pdf_files(cfg)
