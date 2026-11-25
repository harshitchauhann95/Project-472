"""
Baseline Model 3: Fine-Tuned Transformer (DistilBERT / BERT)
Fine-tunes a contextual transformer for complaint category classification.
Provides --fast-dev smoke test support and full held-out test set evaluation.
"""

import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_and_eval_transformer(train_path: str = "data/processed/train.csv",
                               val_path: str = "data/processed/val.csv",
                               test_path: str = "data/processed/test.csv",
                               model_name: str = "distilbert-base-uncased",
                               model_dir: str = "models",
                               results_dir: str = "results",
                               epochs: int = 2,
                               batch_size: int = 16,
                               max_len: int = 128,
                               fast_dev: bool = False):
    """
    Fine-tunes DistilBERT/BERT on the training split and evaluates on the held-out test split.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for Transformer: {device} | Base Model: {model_name}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    if fast_dev:
        print("[FAST-DEV] Running rapid transformer smoke test...")
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(60, len(val_df)), random_state=42)
        test_df = test_df.sample(n=min(60, len(test_df)), random_state=42)
        epochs = 1
        batch_size = 16
        
    classes = sorted(train_df["category"].unique().tolist())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id
    ).to(device)
    
    train_dataset = TransformerDataset(train_df["description"], train_df["category"].map(label2id), tokenizer, max_len)
    val_dataset = TransformerDataset(val_df["description"], val_df["category"].map(label2id), tokenizer, max_len)
    test_dataset = TransformerDataset(test_df["description"], test_df["category"].map(label2id), tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {ep+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
    # Evaluation on held-out test split
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())
            
    y_test_labels = [id2label[i] for i in test_targets]
    y_pred_labels = [id2label[i] for i in test_preds]
    
    acc = float(accuracy_score(y_test_labels, y_pred_labels))
    macro_f1 = float(f1_score(y_test_labels, y_pred_labels, average="macro", zero_division=0))
    rep = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=classes).tolist()
    
    metrics = {
        "model_name": f"Fine-Tuned Transformer ({model_name})",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classes": classes,
        "classification_report": rep,
        "confusion_matrix": cm,
        "test_samples": len(test_df)
    }
    
    # Save tokenizer and metadata
    tokenizer.save_pretrained(os.path.join(model_dir, "transformer_tokenizer"))
    meta = {
        "classes": classes,
        "label2id": label2id,
        "id2label": id2label,
        "max_len": max_len,
        "base_model": model_name
    }
    joblib.dump(meta, os.path.join(model_dir, "transformer_metadata.pkl"))
    
    res_path = os.path.join(results_dir, "transformer_metrics.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n--- Transformer Evaluation ({model_name}) ---")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
    print(f"Metrics saved to {res_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--val", type=str, default="data/processed/val.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fast-dev", action="store_true", help="Quick smoke test flag")
    args = parser.parse_args()
    
    train_and_eval_transformer(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fast_dev=args.fast_dev
    )
