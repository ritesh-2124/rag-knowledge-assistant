from generation.generator import generate_answer


if __name__ == "__main__":

    question = "Explain Kubernetes autoscaling in simple words"

    answer = generate_answer(question)

    print("\nAI Response:\n")
    print(answer)