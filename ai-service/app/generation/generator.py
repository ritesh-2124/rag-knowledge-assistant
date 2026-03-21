import ollama

def generate_answer(query, context_docs):
    context = "\n\n".join(context_docs)

    # Keep prompt short and simple for gemma:2b
    prompt = f"""Read the text below and answer the question.
Extract exact values. Do not say "not found".

Text:
{context}

Question: {query}
Answer:"""

    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']