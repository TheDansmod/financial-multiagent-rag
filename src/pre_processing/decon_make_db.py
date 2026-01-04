r"""Split markdown files, get segment summaries, and populate vector db with the chunks.

First, split the markdown files by the markdown header to obtain segment documents.
Obtain summaries for each segment using an LLM. Then, split the segments sentence by
sentence - using an LLM where needed. Then, decontexualise each sentence by passing the
sentence and some context (previous sentences) to an LLM. Each decontextualised
sentence is now a chunk, as are all of the summaries. Embed both, the sentences and the
summaries into the vector database. Ensure that the summary is part of the metadata of
each sentence.
"""
