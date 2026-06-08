from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
embedding = OllamaEmbeddings(     # Load embedding model load balancing across CPU and GPU very good
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory="./chroma_db",  # chroma db ka path jaha pe humne chunks store kiye hai
    embedding_function=embedding
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

question = input("Ask ur question: ")  
# user ka question input le rahe hai, ye question retriever ko denge, retriever us question ke basis pe relevant chunks ko retrieve karega 
docs = retriever.invoke(question) # chunk created retrived based on user question

context = "\n\n".join(
    [doc.page_content for doc in docs]
)


print("\n RETRIEVED CONTEXT:  \n")
print(context[:1000])


prompt = f"""
You are a helpful assistant.

Answer ONLY from the context provided.

Context:
{context}

Question:
{question}
"""

# Load LLM
llm = ChatOllama(
    model="llama3"
)

response = llm.invoke(prompt)

print("\nAnswer: \n")
print(response.content)