import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


def create_polka_analyst_agent(data_path: str):

    try:
        df_polka = pd.read_csv(data_path)
        print(f"✅ Data upload: {df_polka.shape[0]} verses, {df_polka.shape[1]} columns.")
    except Exception as e:
        raise RuntimeError(f"❌ Error uploading CSV: {e}")

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)


    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df_polka,
        verbose=True,
        agent_type="openai-tools"
    )

    return agent