from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def main():
    # Load environment variables from .env
    load_dotenv()

    # Initialize model
    llm = ChatOpenAI()

    # Invoke model
    response = llm.invoke("Hello")

    print(response.content)


if __name__ == "__main__":
    main()
