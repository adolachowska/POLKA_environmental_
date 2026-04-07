import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from agent_data_analyst import create_polka_analyst_agent

def load_and_clean_data(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path)
    drop_cols = ['country_name', 'modern_country_name', 'year']
    df = df.drop(columns=drop_cols, errors='ignore')

    comma_cols = [
        'soil_ratio', 'landscape_ratio', 'natural_forestation_ratio',
        'max_degree_north', 'max_degree_south', 'border_ratio'
    ]

    for col in comma_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)

    return df


def prepare_features_and_target(df: pd.DataFrame, target_col: str):
    main_cols = ['sub_system_type', 'economic_type', 'if_rich']
    X = df.drop(columns=[target_col] + main_cols, errors='ignore')
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def train_xgboost_model(X_train, y_train):

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        enable_categorical=True,
        random_state=42,
        eval_metric='mlogloss'
    )

    # Train the model
    model.fit(X_train, y_train)
    return model


os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_API_KEY_HERE_OR_USE_DOTENV"


def main():
    print("--- STARTING POLKA SYSTEM ---")

    # 2. Define the path to the data (remember about the relative path!)
    data_path = 'data/environmental_data - Arkusz1.csv'

    # ... You can paste your existing error analysis code (y_test, y_pred, etc.) here ...

    # 3. CREATE THE AGENT using the imported function
    print("\nInitializing AI Agent...")
    polka_agent = create_polka_analyst_agent(data_path)

    # 4. TALKING WITH THE DATA
    query = (
        "Analyze the dataset. What are the three most frequent "
        "soil types (dominant_soil) for countries where 'system_type' is 'Democratic'?"
    )

    print(f"\nQuery to POLKA-Agent: {query}")
    try:
        # The invoke() method runs the agent (older LangChain versions used run())
        response = polka_agent.invoke({"input": query})
        print(f"\nResponse:\n{response['output']}")
    except Exception as e:
        print(f"Error during Agent execution: {e}")

if __name__ == "__main__":

    main()
    file_path = 'data/environmental_data.csv'
    df_data = pd.read_csv(file_path)
    df_clean = load_and_clean_data(file_path)

    target_variable = 'system_type'
    X, y, label_encoder = prepare_features_and_target(df_clean, target_variable)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost Classifier...")
    xgb_model = train_xgboost_model(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    errors = (np.array(y_test) != np.array(y_pred))
    bad_indices = X_test[errors].index
    false_data = df_data.loc[bad_indices, ['country_name', 'year']].copy()
    false_data['actual'] = np.array(y_test)[errors]
    false_data['predicted'] = np.array(y_pred)[errors]

    print(f"\nThe number of wrong prediction: {len(false_data)}")
    print("Countries wrongly predicted:")
    print("-" * 50)
    for _, row in false_data.iterrows():
        print(f"[{row['year']}] {row['country_name']} | "
              f"True: {row['actual']} -> False: {row['predicted']}")

    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
