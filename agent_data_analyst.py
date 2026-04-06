import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


def create_polka_analyst_agent(data_path: str):
    """
    Inicjalizuje Agenta LangChain do analizy danych środowiskowych projektu POLKA.

    Args:
        data_path (str): Ścieżka do pliku CSV z danymi.

    Returns:
        AgentExecutor: Gotowy agent gotowy do przyjmowania zapytań.
    """
    # 1. Wczytanie danych z uwzględnieniem polskiego/europejskiego formatowania CSV
    # (jeśli arkusz używał przecinków, pandas sobie z tym poradzi, ale warto być czujnym)
    try:
        df_polka = pd.read_csv(data_path)
        print(f"✅ Pomyślnie wczytano dane: {df_polka.shape[0]} wierszy, {df_polka.shape[1]} kolumn.")
    except Exception as e:
        raise RuntimeError(f"❌ Błąd wczytywania pliku CSV: {e}")

    # 2. Inicjalizacja LLM (Mózg Agenta) - temperatura 0 dla ścisłych, analitycznych odpowiedzi
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # 3. Utworzenie i zwrócenie agenta Pandas
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df_polka,
        verbose=True,  # Pozwala śledzić tok rozumowania agenta w konsoli PyCharma
        agent_type="openai-tools"
    )

    return agent