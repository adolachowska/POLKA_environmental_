import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


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

def prepare_features_and_target(df: pd.DataFrame, target_col: str, selected_features: list = None):
    main_cols = ['sub_system_type', 'economic_type', 'system_type']

    X = df.drop(columns=[target_col] + main_cols, errors='ignore')
    y = df[target_col]

    if selected_features is not None:
        valid_features = [col for col in selected_features if col in X.columns]
        X = X[valid_features]

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


def save_feature_importance(model, feature_names, output_filename="feature_importance.png"):
    importances = model.get_feature_importance()

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance (%)': importances
    }).sort_values(by='Importance (%)', ascending=False)

    print("\n" + "=" * 50)
    print(f"\n TARGET VARIABLE: {target_variable}")
    print("\n" + "=" * 50)

    print("\n" + "=" * 50)
    print("RANKING OF GEOGRAPHICAL FEATURE IMPORTANCE:")
    print("=" * 50)
    for index, row in fi_df.iterrows():
        print(f"{row['Feature']:<30}: {row['Importance (%)']:.2f}%")

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x='Importance (%)',
        y='Feature',
        data=fi_df,
        palette='viridis',
        hue='Feature',
        legend=False
    )

    plt.title('Impact of Geographical Features on Political System (POLKA - CatBoost)', fontsize=14, pad=15)
    plt.xlabel('Feature Importance (%)', fontsize=12)
    plt.ylabel('Environmental Features', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    print(f"\n[SUCCESS] The chart has been saved as a file: {output_filename}")

if __name__ == "__main__":

    file_path = 'data/environmental_data.csv'

    try:
        df_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        print("Make sure you mapped the volumes correctly in Docker.")
        exit(1)

    df_clean = load_and_clean_data(file_path)

    target_variable = 'if_rich'


    analized_features = [
        'land size',
        'land_size_plus',
        'continent_size',
        'dominant_soil',
        'ore_deposit',
        'dominant_landscape',
        'climate_dominant',
        'climate_t2_dominant'
        'climate_t2_dominant',
        'max_degree_north',
        'max_degree_south'
        #borders:#
        'count_boundaries',
        'longest_border',
        'country_borders',
        'sea_borders',
        'water_borders',
        'desert_borders',
        'mountain_borders',
        'open_borders',
        #ENGINEERED FEATURES:#
        'continent_ratio',
        'soil_ratio',
        'landscape_ratio',
        'border_ratio',
        'natural_forestation_ratio'
    ]

    X, y, label_encoder, cat_features = prepare_features_and_target(
        df_clean,
        target_variable,
        selected_features=analized_features
    )

    #X, y, label_encoder, cat_features = prepare_features_and_target(df_clean, target_variable)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining CatBoost...")
    cat_model = train_catboost_model(X_train, y_train, cat_features)
    save_feature_importance(cat_model, X.columns.tolist(), "feature_importance.png")

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