import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    drop_cols = ['country_name', 'modern_country_name', 'year']
    df = df.drop(columns=drop_cols, errors='ignore')

    comma_cols = [
        #'soil_ratio',
        #'landscape_ratio',
        #'natural_forestation_ratio',
        #'max_degree_north',
        #'max_degree_south',
        #'border_ratio',
        'continent_size'
    ]

    for col in comma_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = (df[col]
                       .astype(str) #string
                       .str.replace('\xa0', '', regex=False) #(\xa0)
                       .str.replace(' ', '', regex=False) #(' ')
                       .str.replace(',', '.', regex=False) #,->.
                       .astype(float)) #float

    return df

def filter_significant_features(df: pd.DataFrame, features: list, target_col: str, alpha: float = 0.05) -> list:
    significant_features = []

    df_valid = df.dropna(subset=[target_col]).copy()

    for col in features:
        if col not in df_valid.columns:
            continue

        p_value = 1.0
        test_name = ""



        if df_valid[col].dtype == 'object' or df_valid[col].dtype.name == 'category':
            test_name = "Chi-Square"
            contingency = pd.crosstab(df_valid[col], df_valid[target_col])
            try:
                chi2, p_value, dof, ex = chi2_contingency(contingency)
            except ValueError:
                p_value = 1.0


        else:
            test_name = "Kruskal-Wallis"
            classes = df_valid[target_col].unique()
            groups = [df_valid[df_valid[target_col] == c][col].dropna() for c in classes]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) > 1:
                try:
                    stat, p_value = kruskal(*groups)
                except ValueError:
                    p_value = 1.0

        status = "OK" if p_value < alpha else "CATEGORY WITH NO IMPACT"

        if p_value < alpha:
            significant_features.append(col)

        print(f"{status:22} | {col:<25} | Test: {test_name:<14} | p-value: {p_value:.4f}")

    print(
        f" {len(significant_features)} from {len(features)} is significant")
    for col in significant_features:
        print(col)

    return significant_features


def prepare_features_and_target(df: pd.DataFrame, target_col: str, selected_features: list = None):
    #main_cols = ['if_rich', 'sub_system_type', 'system_type']
    main_cols=['if_rich']

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


def plot_feature_to_target_heatmap(df, feature_col, target_col, output_filename="feature_target_heatmap.png"):

    print(f"\nAnalyzing relationship between '{feature_col}' and '{target_col}'...")

    # Calculate the cross-tabulation (converting counts to percentages per row)
    cross_tab = pd.crosstab(df[feature_col], df[target_col], normalize='index') * 100

    # Generate the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cross_tab,
        annot=True,  # Show the exact percentages
        fmt=".1f",  # Format to 1 decimal place
        cmap="YlGnBu",  # Color scale (Yellow-Green-Blue)
        cbar_kws={'label': 'Probability (%)'}
    )

    plt.title(f'How {feature_col} influences {target_col} (POLKA)', fontsize=14, pad=15)
    plt.xlabel(f'Predicted System ({target_col})', fontsize=12)
    plt.ylabel(f'Geographical Feature ({feature_col})', fontsize=12)

    # Rotate the y-axis labels for better readability if text is long
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_filename, dpi=300)
    print(f"[SUCCESS] Heatmap saved as: {output_filename}")


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

    file_path = 'data/environmental_data_3.2.csv'

    try:
        df_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        print("Make sure you mapped the volumes correctly in Docker.")
        exit(1)

    df_clean = load_and_clean_data(file_path)

    target_variable = 'system_type'
    feature_to_analyze = 'continent_ratio'


    analized_features = [

        #'year_from',
        'land_size',
        #'land_size_plus',
        'continent_size',
        #'capital_location',
        #'dominant_soil',
        #'ore_deposit',
        #'dominant_landscape',
        #'climate_dominant',
        #'climate_t2_dominant'
        #'climate_t2_dominant',
        #'max_degree_north',
        #'max_degree_south'
        #borders:
        #'count_boundaries',
        #'longest_border',
        #'country_borders',
        #'sea_borders',
        #'water_borders',
        #'desert_borders',
        #'mountain_borders',
        #'open_borders',
        #ENGINEERED FEATURES:
        'continent_ratio'
        #'soil_ratio',
        #'landscape_ratio',
        #'border_ratio',
        #'natural_forestation_ratio'
    ]

    plot_feature_to_target_heatmap(
        df=df_clean,
        feature_col=feature_to_analyze,
        target_col=target_variable,
        output_filename=f"heatmap_{feature_to_analyze}_vs_{target_variable}.png"
    )

    statistically_valid_features = filter_significant_features(
        df=df_clean,
        features=analized_features,
        target_col=target_variable,
        alpha=0.05  #trust 0.05
    )

    if not statistically_valid_features:
        statistically_valid_features = analized_features

    X, y, label_encoder, cat_features = prepare_features_and_target(
        df_clean,
        target_variable,
        selected_features=statistically_valid_features
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