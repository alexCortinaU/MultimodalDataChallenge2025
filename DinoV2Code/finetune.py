from train import Config
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

def create_embedding_map(tokens, n_components=12):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(tokens)
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(embeddings)
    embedding_map = dict(zip(tokens, embeddings))
    return embedding_map


def train_random_forest(logits_path):
    config = Config()
    df_meta = pd.read_csv(config.metadata_dir)

    df_train = df_meta[df_meta['filename_index'].str.startswith('fungi_train')]
    df_train.loc[:, 'taxonID_index'] = df_train['taxonID_index'].astype(int)

    df_logits = pd.read_csv(logits_path, skiprows=1)
    filename_to_label = dict(zip(df_train['filename_index'], df_train['taxonID_index']))

    mask = df_logits['filename'].isin(filename_to_label.keys())
    df_logits_filtered = df_logits[mask].copy()
    y = df_logits_filtered['filename'].map(filename_to_label).values

    # Features (all columns except filename)
    x = df_logits_filtered.iloc[:, 1:].values

    # Train-validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=config.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=config.seed,
        class_weight='balanced'  # Handle class imbalance
    )
    clf.fit(x_train, y_train)

    # Evaluate
    y_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    # Feature importance
    importance = clf.feature_importances_
    print(f"Top 5 feature indices: {np.argsort(importance)[-5:][::-1]}")

    # Save model
    joblib.dump(clf, "random_forest_model.joblib")
    print("Model saved to random_forest_model.joblib")

    return clf, acc


if __name__ == "__main__":
    train_random_forest('vit_v1_test_logits.csv')
