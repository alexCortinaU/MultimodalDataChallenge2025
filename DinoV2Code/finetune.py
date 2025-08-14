from train import Config
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
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


def train_random_forest(logits_path, output_path):
    config = Config()
    df_meta = pd.read_csv(config.metadata_dir)

    df_train = df_meta[df_meta['filename_index'].str.startswith('fungi_train')].reset_index()
    df_test = df_meta[df_meta['filename_index'].str.startswith('fungi_test')].reset_index()
    #habitat_emb = create_embedding_map(df_train['Habitat'])
    #substrate_emb = create_embedding_map(df_train['Habitat'])

    df_train['taxonID_index'] = df_train['taxonID_index'].astype(int)

    df_logits = pd.read_csv(logits_path, skiprows=1)
    df_logits_train = df_logits[df_logits['filename'].str.startswith('fungi_train')].reset_index()
    df_logits_test = df_logits[df_logits['filename'].str.startswith('fungi_test')].reset_index()

    df_logits_train = df_logits_train.iloc[:, 2:]

    print(df_train['taxonID_index'].head())
    print(df_logits_train.head())
    x_train, x_val, y_train, y_val = train_test_split(
        df_logits_train,  df_train['taxonID_index'].values, test_size=0.2, random_state=config.seed,
    )

    # Hyperparameter space
    param_dist = {
        'n_estimators': [200, 500, 800, 1200],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    clf = RandomForestClassifier(
        n_jobs=-1,
        random_state=config.seed,
        class_weight='balanced'
    )

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=30,
        scoring=make_scorer(f1_score, average='macro'),
        cv=3,
        verbose=1,
        random_state=config.seed,
        n_jobs=-1
    )

    search.fit(x_train, y_train)

    print("Best parameters:", search.best_params_)
    print("Best CV macro F1:", search.best_score_)
    clf.fit(x_train, y_train)

    # Evaluate
    y_pred = clf.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    filenames = df_logits_test['filename'].values
    predictions = clf.predict(df_logits_test.iloc[:, 2:])

    # Save predictions
    with open(f'{output_path}.csv', 'w') as f:
        f.write('vit_is_all_you_need\n')
        for filename, pred in zip(filenames, predictions):
            f.write(f'{filename},{pred}\n')

    print(f"Predictions saved to {output_path}")
    print(f"Generated {len(predictions)} predictions for test set")
    return output_path

if __name__ == "__main__":
    train_random_forest('vit_v1_test_logits.csv', 'vit_rf')
