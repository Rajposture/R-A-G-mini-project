from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import os 
import shutil

pdf_path = "./data/Retrieval Augmented Generation guide.pdf"
pdf_path = "./data/ai.pdf"  ## absolute path to the pdf file
pdf_path = "./data/ml.pdf"  ## relative path to the pdf file, assuming the script is run from the project root

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db") # purana database delete krne ke liye, taki naya data add ho sake

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

filename = os.path.basename(pdf_path) #os interfair to get the filename from the path 

for chunk in chunks:
    chunk.metadata["source"] = filename

embedding = OllamaEmbeddings(
    model="nomic-embed-text" # Use the same embedding model as in app.py for consistency
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

print(f"Stored {len(chunks)} chunks")