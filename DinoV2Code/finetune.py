from train import Config
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

def create_embedding_map(tokens, output_path, n_components=12):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(tokens)
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(embeddings)
    embedding_map = dict(zip(tokens, embeddings))
    return embedding_map


def train_random_forest(logits_path, output_path):
    config = Config()
    df_meta = pd.read_csv(config.metadata_dir)

    df_train = df_meta[df_meta['filename_index'].str.startswith('fungi_train')].reset_index()
    df_test = df_meta[df_meta['filename_index'].str.startswith('fungi_test')].reset_index()

    df_train['taxonID_index'] = df_train['taxonID_index'].astype(int)

    df_logits = pd.read_csv(logits_path, skiprows=1)
    df_logits_train = df_logits[df_logits['filename'].str.startswith('fungi_train')].reset_index()
    df_logits_test = df_logits[df_logits['filename'].str.startswith('fungi_test')].reset_index()

    print("First 10 logits filenames:")
    print(df_logits_train['filename'].head(10))

    print("\nFirst 10 metadata filenames:")
    print(df_train['filename_index'].head(10))

    print(df_logits_train.filename)
    print(df_train.filename_index)

    x_train, x_val, y_train, y_val = train_test_split(
        df_logits_train, y, test_size=0.2, random_state=config.seed, stratify=y
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

    with open(output_path, 'w') as f:
        f.write('random_fungi_predictions\n')  # Description line
        for filename, pred in zip(filenames, predictions):
            f.write(f'{filename},{pred}\n')

    print(f"Predictions saved to {output_path}")
    print(f"Generated {len(predictions)} predictions for test set")
    return output_path

if __name__ == "__main__":
    train_random_forest('vit_v1_test_logits.csv', 'return_of_the_vit')
