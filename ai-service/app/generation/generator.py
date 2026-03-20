import ollama

def generate_answer(query, context_docs):
    context = "\n".join(context_docs)

    prompt = f"""
You are an AI assistant.

Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']