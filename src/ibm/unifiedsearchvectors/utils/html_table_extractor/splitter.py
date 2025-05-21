#%%
import os
import pickle
import re

import yaml
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import (
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
)

print(os.getcwd())
with open('../src/ibm/unifiedsearchvectors/utils/html_table_extractor/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

docs_path = config["paths"]["output_folder"]

loader_txt = DirectoryLoader(
        docs_path,  # documents folder
        glob="**/*.txt",  # we only get TXTs
        show_progress=True,
    )

print("Loading TXT tables...")
docs_txt = loader_txt.load()
print("Loaded TXT tables...")
print("Number of TXT files loaded: ", len(docs_txt))

text_meta_dict = {}
for tbl in docs_txt:
    metadata, content = tbl.page_content.split("Table Content:")
    text_meta_dict[tbl.metadata['source']] = metadata
    tbl.page_content = content

# Parameters to tune
chunk_size = 512
chunk_overlap = 128
model_name = 'thenlper/gte-small'
text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=chunk_size, model_name=model_name, chunk_overlap=chunk_overlap
)

docs_split = text_splitter.split_documents(docs_txt)

print("Documents split finished...")
print("Number of splits: ", len(docs_split))

for doc in docs_split:
    doc.page_content = text_meta_dict[doc.metadata['source']] + doc.page_content
print(docs_split)

# Saving the list of document chunks to a pickle file.
# These can then be embedded using your favourite embedder.
# with open(f'{docs_path}/splits.pkl', 'wb') as handle:
#     pickle.dump(docs_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%