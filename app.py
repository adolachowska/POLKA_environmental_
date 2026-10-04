import pandas as pd
import sys
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOllama


def run_polka_free_eda_agent(file_path: str):

    print("\n" + "=" * 50)
    print("POLKA SMART EDA AGENT (Free Local Model)")
    print("=" * 50)

    try:
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Loaded data: {file_path}")
        print(f"[INFO] Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        print("Ensure you mapped your volumes correctly in Docker using -v.")
        sys.exit(1)


    print("\nConnecting to the local Llama 3.1 model...")
    try:
        llm = ChatOllama(
            model="llama3.1",
            base_url="http://host.docker.internal:11434",
            temperature=0 #->determinism
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True,
            agent_type="zero-shot-react-description"
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to the model: {e}")
        sys.exit(1)

    print("\nAgent is ready! Ask your geographical and political questions.")
    print("Example: 'Which countries have a Federal system_type and mountainous dominant_landscape?'")
    print("Type 'exit', 'quit', or 'q' to close the assistant.\n")

    # Interactive chat loop
    while True:
        try:
            user_question = input("POLKA> ")
        except (KeyboardInterrupt, EOFError):
            print("\nClosing EDA Agent. Goodbye!")
            break

        if user_question.lower() in ['exit', 'quit', 'q']:
            print("Closing EDA Agent. Goodbye!")
            break

        if not user_question.strip():
            continue

        try:
            response = agent.invoke(user_question)
            print(f"\n[Agent]: {response['output']}\n")
        except Exception as e:
            print(f"\n[Analysis Error]: An issue occurred while processing your request: {e}\n")


if __name__ == "__main__":
    # path inside the Docker container
    target_file_path = 'data/environmental_data.csv'
    run_polka_free_eda_agent(target_file_path)

    #docker run -it --rm -v "${PWD}:/app" polka-ml-env python app.py#