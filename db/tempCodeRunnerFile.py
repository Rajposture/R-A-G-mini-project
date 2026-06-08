

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