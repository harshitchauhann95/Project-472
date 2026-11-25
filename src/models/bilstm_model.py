"""
Baseline Model 2: PyTorch Bidirectional LSTM (BiLSTM)
Implements an end-to-end word-level BiLSTM classifier with vocab indexer,
MPS/CPU device handling, --fast-dev smoke test support, and rigorous evaluation.
"""

import os
import re
import json
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


class HostelTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        seq = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens][:self.max_len]
        
        # Padding
        if len(seq) < self.max_len:
            seq = seq + [self.vocab["<pad>"]] * (self.max_len - len(seq))
            
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=5, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, (hn, cn) = self.lstm(emb)
        # Global max pooling over sequence length
        pooled, _ = torch.max(out, dim=1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits


def build_vocab(texts, min_freq=1):
    freq = {}
    for t in texts:
        tokens = [w.lower() for w in re.findall(r'\b\w+\b', str(t))]
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
            
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in freq.items():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def train_and_eval_bilstm(train_path: str = "data/processed/train.csv",
                          val_path: str = "data/processed/val.csv",
                          test_path: str = "data/processed/test.csv",
                          model_dir: str = "models",
                          results_dir: str = "results",
                          epochs: int = 5,
                          batch_size: int = 32,
                          max_len: int = 128,
                          fast_dev: bool = False):
    """
    Trains BiLSTM model on train split, validates on val, and evaluates on test.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for BiLSTM: {device}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    if fast_dev:
        print("[FAST-DEV] Running small-subset smoke test (epochs=2)...")
        train_df = train_df.sample(n=min(300, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(100, len(val_df)), random_state=42)
        test_df = test_df.sample(n=min(100, len(test_df)), random_state=42)
        epochs = 2
        batch_size = 16
        
    classes = sorted(train_df["category"].unique().tolist())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}
    
    vocab = build_vocab(train_df["description"])
    print(f"Vocabulary size: {len(vocab)} words | Classes: {len(classes)}")
    
    train_dataset = HostelTextDataset(train_df["description"], train_df["category"].map(label2id), vocab, max_len)
    val_dataset = HostelTextDataset(val_df["description"], val_df["category"].map(label2id), vocab, max_len)
    test_dataset = HostelTextDataset(test_df["description"], test_df["category"].map(label2id), vocab, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        num_classes=len(classes),
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    best_val_f1 = 0.0
    best_weights_path = os.path.join(model_dir, "bilstm_model.pt")
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                logits = model(x_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_b.numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {ep+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Macro-F1: {val_f1:.4f}")
        
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_weights_path)
            
    # Evaluation on held-out test split
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            test_preds.extend(preds)
            test_targets.extend(y_b.numpy())
            test_probs.extend(probs)
            
    y_test_labels = [id2label[i] for i in test_targets]
    y_pred_labels = [id2label[i] for i in test_preds]
    
    acc = float(accuracy_score(y_test_labels, y_pred_labels))
    macro_f1 = float(f1_score(y_test_labels, y_pred_labels, average="macro", zero_division=0))
    rep = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=classes).tolist()
    
    metrics = {
        "model_name": "PyTorch BiLSTM Neural Network",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classes": classes,
        "classification_report": rep,
        "confusion_matrix": cm,
        "test_samples": len(test_df)
    }
    
    # Save metadata & vocab
    meta = {
        "classes": classes,
        "label2id": label2id,
        "id2label": id2label,
        "vocab": vocab,
        "max_len": max_len,
        "device": str(device)
    }
    joblib.dump(meta, os.path.join(model_dir, "bilstm_metadata.pkl"))
    
    res_path = os.path.join(results_dir, "bilstm_metrics.json")
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n--- BiLSTM Neural Network Evaluation ---")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
    print(f"Metrics saved to {res_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--val", type=str, default="data/processed/val.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fast-dev", action="store_true", help="Quick smoke test flag")
    args = parser.parse_args()
    
    train_and_eval_bilstm(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fast_dev=args.fast_dev
    )
