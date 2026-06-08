from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("./data/Retrieval Augmented Generation guide.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="./chroma_db"
)

print(f"Stored {len(chunks)} chunks")