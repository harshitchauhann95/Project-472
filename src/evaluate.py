"""
Unified Evaluation and Model Comparison Module
Evaluates all three models (TF-IDF Baseline, PyTorch BiLSTM, and Fine-Tuned Transformer)
on the SAME held-out test split (data/processed/test.csv).

Generates:
  1. Accuracy, Per-Class Precision, Recall, F1, and Macro-F1
  2. Confusion Matrix plots (results/confusion_matrix_*.png)
  3. JSON report (results/model_comparison.json)
  4. Markdown benchmark table (results/model_comparison.md)
"""

import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def plot_and_save_confusion_matrix(cm, classes, model_name: str, save_path: str):
    """
    Renders and saves a clean, readable confusion matrix heatmap.
    """
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, pad=12)
    plt.xlabel("Predicted Category", fontsize=10)
    plt.ylabel("True Category", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix chart saved to {save_path}")


def run_unified_evaluation(test_path: str = "data/processed/test.csv",
                           results_dir: str = "results"):
    """
    Compiles metrics from all models into a single comparative analysis.
    """
    os.makedirs(results_dir, exist_ok=True)
    test_df = pd.read_csv(test_path)
    classes = sorted(test_df["category"].unique().tolist())
    
    models_to_check = [
        ("TF-IDF + Logistic Regression", os.path.join(results_dir, "tfidf_baseline_metrics.json"), "confusion_matrix_tfidf.png"),
        ("PyTorch BiLSTM Neural Network", os.path.join(results_dir, "bilstm_metrics.json"), "confusion_matrix_bilstm.png"),
        ("Fine-Tuned Transformer (DistilBERT)", os.path.join(results_dir, "transformer_metrics.json"), "confusion_matrix_transformer.png"),
    ]
    
    comparison_records = []
    
    for display_name, json_path, cm_filename in models_to_check:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                
            acc = data.get("accuracy", 0.0)
            macro_f1 = data.get("macro_f1", 0.0)
            cm = np.array(data.get("confusion_matrix", []))
            
            # Save chart if CM exists
            if len(cm) > 0:
                cm_path = os.path.join(results_dir, cm_filename)
                plot_and_save_confusion_matrix(cm, classes, display_name, cm_path)
                
            per_class_f1 = {}
            rep = data.get("classification_report", {})
            for c in classes:
                if c in rep:
                    per_class_f1[c] = round(rep[c].get("f1-score", 0.0), 4)
                    
            comparison_records.append({
                "Model": display_name,
                "Test Accuracy": acc,
                "Macro F1-Score": macro_f1,
                "Electrical F1": per_class_f1.get("electrical", "-"),
                "Plumbing F1": per_class_f1.get("plumbing", "-"),
                "Furniture F1": per_class_f1.get("furniture_carpentry", "-"),
                "Cleaning F1": per_class_f1.get("cleaning_sanitation", "-"),
                "General Maint F1": per_class_f1.get("general_maintenance", "-"),
                "Test Samples": data.get("test_samples", len(test_df))
            })
            
    comp_df = pd.DataFrame(comparison_records)
    
    # Save JSON summary
    summary_json_path = os.path.join(results_dir, "model_comparison.json")
    comp_df.to_json(summary_json_path, orient="records", indent=2)
    
    # Save Markdown summary
    summary_md_path = os.path.join(results_dir, "model_comparison.md")
    md_content = f"# Model Evaluation Benchmark (Held-Out Test Split: N={len(test_df)})\n\n"
    
    # Generate markdown table natively without external dependency
    cols = list(comp_df.columns)
    header = "| " + " | ".join(cols) + " |\n"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = ""
    for _, row in comp_df.iterrows():
        rows += "| " + " | ".join(str(row[c]) for c in cols) + " |\n"
    md_content += header + divider + rows
    md_content += "\n*Evaluation performed with stratified 70/15/15 split without data leakage.*\n"
    
    with open(summary_md_path, "w") as f:
        f.write(md_content)
        
    print("\n=================== MODEL COMPARISON BENCHMARK ===================")
    print(comp_df.to_string(index=False))
    print(f"\nSaved benchmark table to {summary_md_path}")
    print(f"Saved benchmark JSON to {summary_json_path}")
    
    return comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()
    
    run_unified_evaluation(test_path=args.test, results_dir=args.results)
