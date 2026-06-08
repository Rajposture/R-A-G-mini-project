from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
app = FastAPI()
embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

llm = ChatOllama(
    model="llama3"
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: Question): # user question form api

    docs = retriever.invoke(data.question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    prompt = f"""
    Answer only using the provided context.
    Context:
    {context}

    Question:
    {data.question}
    """

    response = llm.invoke(prompt)

    return {
        "question": data.question,
        "answer": response.content,
        "sources": len(docs)
    }