import ollama


def generate_answer(question: str):

    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]