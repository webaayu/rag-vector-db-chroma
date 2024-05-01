import streamlit as st
import os
import pandas as pd
import numpy as np
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./data.txt")
document=loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(document)
print(texts[0].page_content)
print(texts[1])

#create DB
persist_directory = 'db'
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(documents=texts,embedding=embeddings,persist_directory=persist_directory)
print("Embedding is written in binary file")
#persist data to the disk
vectordb.persist()
vectordb=None
#load the persist db from the disk and use it as normal
vectordb=Chroma(persist_directory=persist_directory,embedding_function=embeddings)
print(vectordb)
#Make a retriever
retriever=vectordb.as_retriever()
#query="What is alergy?"
#print(query)
prompt=st.text_input("Enter your Question:")
docs=retriever.get_relevant_documents(prompt)
st.write(docs)
