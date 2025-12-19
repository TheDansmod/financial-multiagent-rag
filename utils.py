def get_sec_10k_data(data_folder, companies):
    # data_folder = "/mnt/windows/Users/lordh/Documents/Svalbard/Data"
    # companies = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "WMT", "JPM", "NFLX"]
    from sec_edgar_downloader import Downloader
    dl = Downloader("Google", "vke4@gmail.com", data_folder)
    for company in companies:
        try:
            dl.get("10-K", company, limit=1, download_details=False)
            print(f"Downloaded 10-K for {company}")
        except Exception as e:
            print(f"Failed to download {company}: {e}")

def clean_and_convert_to_markdown():
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    import re

    data_folder = r"data/sec_filings/"
    companies = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "WMT", "JPM", "NFLX"]
    for company in companies:
        with open(f"{data_folder}{company}-full-submission.txt", "r") as fr, open(f"{data_folder}{company}-parsed.md", "w") as fw:
            raw_content = fr.read()
            soup = BeautifulSoup(raw_content, 'html.parser')
            for script in soup(["script", "style", "head", "title", "meta"]):
                script.extract()

            # 3. Convert to Markdown (Preserves tables & headers)
            markdown_text = md(str(soup), heading_style="ATX")

            # 4. Clean up excessive newlines created by conversion
            markdown_text = re.sub(r'\n\s*\n', '\n\n', markdown_text)
            fw.write(markdown_text)

def convert_sec_filing_to_markdown(src_filepath, dst_filepath):
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    import re

    with open(src_filepath, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    # 1. Regex to find all <DOCUMENT> blocks
    # SEC files structure: <DOCUMENT>\n<TYPE>10-K...\n<TEXT>...html content...</TEXT>\n</DOCUMENT>
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')

    type_pattern = re.compile(r'<TYPE>(.*?)[\n\r]')

    # Find start and end positions of all documents
    starts = [m.start() for m in doc_start_pattern.finditer(raw_content)]
    ends = [m.end() for m in doc_end_pattern.finditer(raw_content)]

    target_document = None

    for start, end in zip(starts, ends):
        doc_content = raw_content[start:end]

        # Check the document TYPE
        type_match = type_pattern.search(doc_content)
        if type_match:
            doc_type = type_match.group(1).strip()

            # We only want the main 10-K filing (sometimes labeled 10-K/A for amendments)
            if doc_type == '10-K':
                target_document = doc_content
                break

    if not target_document:
        return "Error: 10-K document not found in the file."

    # 2. Extract the HTML content specifically between <TEXT> tags
    text_content_match = re.search(r'<TEXT>(.*?)</TEXT>', target_document, re.DOTALL)  # matches newline as well
    if not text_content_match:
        return "Error: No <TEXT> content found in the 10-K document."

    html_content = text_content_match.group(1)

    # 3. Use BeautifulSoup to handle the specific HTML (Cleaning)
    # This prevents the 'zip' error because we excluded the binary documents
    soup = BeautifulSoup(html_content, 'lxml') # lxml is faster for large files

    # Optional: Remove XBRL tags or hidden tables if they clutter the markdown
    for tag in soup.find_all(['xml', 'type']):
        tag.decompose()

    # 4. Convert to Markdown
    # strip=['a'] removes links to keep it clean for RAG, keep them if you need sources
    markdown_text = md(str(soup), heading_style="ATX", strip=['img', 'a'])

    # Post-processing to clean up excessive newlines common in SEC conversions
    markdown_text = re.sub(r'\n\s*\n', '\n\n', markdown_text)

    with open(dst_filepath, "w") as f:
        f.write(markdown_text)

def get_markdown_sec_filings(ticker, dest_filepath):
    # data_folder = r"data/edgar_tools_filings/"
    # companies = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "WMT", "JPM", "NFLX"]
    # for company in companies:
    #     print(company)
    #     src_filepath = f"{data_folder}{company}-full-submission.txt"
    #     dst_filepath = f"{data_folder}{company}.md"
    #     get_markdown_sec_filings(company, dst_filepath)
    from edgar import set_identity, Company
    set_identity("Vipin Kumar vipinkumar1993@gmail.com")
    company = Company(ticker)
    filing = company.get_filings(form="10-K").latest()
    text = filing.markdown()
    with open(dest_filepath, 'w') as f:
        f.write(text)

def clean_edgar_markdown(src_filepath, dst_filepath):
    # data_folder = r"data/edgar_tools_filings/"
    # companies = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "WMT", "JPM", "NFLX"]
    # for company in companies:
    #     print(company)
    #     src_filepath = f"{data_folder}{company}.md"
    #     dst_filepath = f"{data_folder}{company}-cleaned.md"
    #     clean_edgar_markdown(src_filepath, dst_filepath)
    import re
    with open(src_filepath, 'r') as f:
        content = f.read()
    pattern = re.compile(r'<div[^>]*>|</div>', flags=re.IGNORECASE)
    cleaned_content = pattern.sub('', content)
    with open(dst_filepath, 'w') as f:
        f.write(cleaned_content)

def generate_qdrant_db(src_file, company):
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    with open(src_file, 'r') as f:
        text = f.read()
    header_splits = md_splitter.split_text(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    splits = text_splitter.split_documents(md_header_splits)
    for split in splits:
        split.metadata['company'] = company
        split.metadata['year'] = "2025"

    # get embedding for the splits and put them in the db
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=client.get_embedding_size(EMBEDDING_MODEL), distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Appending documents...")
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    vector_store.add_documents(documents=docs, ids=[str(uuid4()) for _ in range(len(docs))])

if __name__ == '__main__':
    data_folder = r"data/edgar_tools_filings/"
    companies = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "WMT", "JPM", "NFLX"]
    for company in companies:
        print(company)
        src_filepath = f"{data_folder}{company}.md"
        dst_filepath = f"{data_folder}{company}-cleaned.md"
        clean_edgar_markdown(src_filepath, dst_filepath)
