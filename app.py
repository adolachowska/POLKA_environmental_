import pandas as pd
import sys
import warnings
import requests

#no warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

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
        sys.exit(1)

    # --- NOWY TEST POŁĄCZENIA ---
    print("\n[DIAGNOSTICS] Testing the physical connection Docker -> Windows...")
    try:
        # Attempt to reach Ollama. If it does not respond within 5 seconds, an error will be raised.
        test_response = requests.get("http://host.docker.internal:11434/api/tags", timeout=5)
        test_response.raise_for_status()
        print("[OK] Connection to Ollama on Windows is working correctly!")
    except requests.exceptions.RequestException as e:
        print(f"\n[CRITICAL NETWORK ERROR] Docker cannot connect to Ollama.")
        print("Make sure you have set the OLLAMA_HOST=0.0.0.0 environment variable on Windows and restarted Ollama.")
        print(f"Technical details: {e}")
        sys.exit(1)
    # ----------------------------

    print("\nConnecting to the local Llama 3.1 model...")
    try:
        llm = ChatOllama(
            model="llama3.1",
            base_url="http://host.docker.internal:11434",
            temperature=0
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,  # ZMIANA NA TRUE: Zobaczymy na zielono proces pisania kodu przez AI
            allow_dangerous_code=True,
            agent_type="zero-shot-react-description"
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to the model: {e}")
        sys.exit(1)

    print("\nAgent is ready! Ask your geographical and political questions.")

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
            print(
            response = agent.invoke(user_question)
            print(f"\n[Agent]: {response['output']}\n")
        except Exception as e:
            print(f"\n[Analysis Error]: An issue occurred while processing your request: {e}\n")


if __name__ == "__main__":
    target_file_path = 'data/environmental_data.csv'
    run_polka_free_eda_agent(target_file_path)

    #docker run -it --rm -v "${PWD}:/app" polka-ml-env python app.py#