from transformers import pipeline
from ddgs import DDGS

print("===================================")
print("   AI Research Assistant Started   ")
print("===================================\n")

print("Loading AI model... Please wait.\n")

# Load AI Model
chatbot = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

print("AI Assistant Ready!")
print("Type 'exit' to stop.\n")

while True:

    # User Input
    user_question = input("You: ")

    # Exit Condition
    if user_question.lower() == "exit":
        print("\nGoodbye!")
        break

    print("\nSearching the internet...\n")

    try:

        search_text = ""

        # Internet Search
        with DDGS() as ddgs:

            results = ddgs.text(
                user_question,
                max_results=5
            )

            for r in results:

                # Collect title
                if "title" in r:
                    search_text += r["title"] + "\n"

                # Collect body/content
                if "body" in r:
                    search_text += r["body"] + "\n"

        # If no results
        if search_text.strip() == "":
            print("No information found.\n")
            continue

        # Prompt for AI
        prompt = f"""
        You are an intelligent AI research assistant.

        Use the information below to answer accurately.

        Information:
        {search_text}

        Question:
        {user_question}

        Give a clear, correct, and short answer.
        """

        # Generate AI Response
        response = chatbot(
            prompt,
            max_length=200,
            do_sample=False
        )

        answer = response[0]["generated_text"]

        print("===================================")
        print("AI Answer:\n")
        print(answer)
        print("===================================\n")

    except Exception as e:

        print("Something went wrong.")
        print("Error:", e)
        print()