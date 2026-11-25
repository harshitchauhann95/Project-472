"""
Baseline Model 1: TF-IDF + Classifier
Trains and evaluates a transparent TF-IDF keyword baseline on the held-out test split.
"""

import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def train_and_eval_tfidf(train_path: str = "data/processed/train.csv",
                         test_path: str = "data/processed/test.csv",
                         model_dir: str = "models",
                         results_dir: str = "results",
                         fast_dev: bool = False):
    """
    Trains TF-IDF + LogisticRegression baseline and evaluates on test set.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    if fast_dev:
        print("[FAST-DEV] Subsetting training data for rapid smoke test...")
        train_df = train_df.sample(n=min(300, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(100, len(test_df)), random_state=42)
        
    X_train = train_df["description"].astype(str)
    y_train = train_df["category"].astype(str)
    X_test = test_df["description"].astype(str)
    y_test = test_df["category"].astype(str)
    
    classes = sorted(y_train.unique().tolist())
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        token_pattern=r'(?u)\b\w+\b'  # keeps single tokens like 'no'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.5)
    clf.fit(X_train_vec, y_train)
    
    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)
    
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    
    metrics = {
        "model_name": "TF-IDF + LogisticRegression Baseline",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classes": classes,
        "classification_report": rep,
        "confusion_matrix": cm,
        "test_samples": len(test_df)
    }
    
    # Save artifacts
    joblib.dump(clf, os.path.join(model_dir, "tfidf_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    
    res_path = os.path.join(results_dir, "tfidf_baseline_metrics.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n--- TF-IDF Baseline Evaluation ---")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Metrics saved to {res_path}")
    
    return metrics, clf, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--fast-dev", action="store_true", help="Quick smoke test flag")
    args = parser.parse_args()
    
    train_and_eval_tfidf(train_path=args.train, test_path=args.test, fast_dev=args.fast_dev)
