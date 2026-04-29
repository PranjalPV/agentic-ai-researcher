from dotenv import load_dotenv
load_dotenv()

from crew import crew

if __name__ == "__main__":
    topic = input("Enter research topic: ")
    result = crew.kickoff(inputs={"query": topic})
    print(result)



# python -m venv venv
# venv\Scripts\activate