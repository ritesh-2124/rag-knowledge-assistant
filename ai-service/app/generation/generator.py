import ollama

def generate_answer(query, context_docs):
    context = "\n\n".join(context_docs)

    prompt = f"""
You are an intelligent AI assistant.

Use ONLY the provided context to answer the question.

Guidelines:
- Carefully read the context
- Identify important entities (names, topics, concepts)
- Answer accurately based on the content
- If the answer is not present, say "Not found in document"

Context:
{context}

Question:
{query}

Answer clearly and completely:
"""

    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']