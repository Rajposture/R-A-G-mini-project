from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("./data/Retrieval Augmented Generation guide.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50  # har chunk ke beech me 50 characters ka overlap hoga, taki context loss na ho jab chunks create ho rahe hai
)

chunks = splitter.split_documents(documents)

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma.from_documents(
    chunks,
    embedding, # embedding function provide karna hoga taki ye chunks ko vector me convert kar sake
    persist_directory="./chroma_db"
)

print(f"Stored {len(chunks)} chunks")
#phase 1 completed