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
def ask_question(data: Question):

    print("=" * 50)
    print("Question:", data.question)

    print("Starting retrieval...")
    docs = retriever.invoke(data.question)
    print(f"Retrieved {len(docs)} docs")

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    print("Context length:", len(context))

    prompt = f"""
    Answer only using the provided context.

    Context:
    {context}

    Question:
    {data.question}
    """

    print("Sending prompt to ollama phi3:mini...")

    response = llm.invoke(prompt)

    print("Response received!")

    return {
        "question": data.question,
        "answer": response.content,
        "sources": len(docs)
    }