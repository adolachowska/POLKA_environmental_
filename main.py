import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


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

    categorical_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].fillna("Missing").astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le, categorical_cols


def train_catboost_model(X_train, y_train, cat_features):
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        random_seed=42,
        loss_function='MultiClass',
        verbose=50
    )

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":

    file_path = 'data/environmental_data.csv'

    try:
        df_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        print("Make sure you mapped the volumes correctly in Docker.")
        exit(1)

    df_clean = load_and_clean_data(file_path)

    target_variable = 'system_type'
    X, y, label_encoder, cat_features = prepare_features_and_target(df_clean, target_variable)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining CatBoost...")
    cat_model = train_catboost_model(X_train, y_train, cat_features)

    # Ewaluacja
    y_pred = cat_model.predict(X_test)
    y_pred = y_pred.flatten()

    accuracy = accuracy_score(y_test, y_pred)


    errors = (np.array(y_test) != y_pred)
    bad_indices = X_test[errors].index


    false_data = df_data.loc[bad_indices, ['country_name', 'year']].copy()


    false_data['actual'] = label_encoder.inverse_transform(np.array(y_test)[errors])
    false_data['predicted'] = label_encoder.inverse_transform(y_pred[errors])

    print(f"\nThe number of wrongly predicted countries: {len(false_data)}")
    print("The countries wrongly predicted:")
    print("-" * 50)
    for _, row in false_data.iterrows():
        print(f"[{row['year']}] {row['country_name']} | "
              f"True: {row['actual']} -> False: {row['predicted']}")

    print(f"\n (Accuracy): {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))