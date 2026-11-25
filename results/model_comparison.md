# Model Evaluation Benchmark (Held-Out Test Split: N=450)

| Model | Test Accuracy | Macro F1-Score | Electrical F1 | Plumbing F1 | Furniture F1 | Cleaning F1 | General Maint F1 | Test Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 450 |
| PyTorch BiLSTM Neural Network | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 450 |
| Fine-Tuned Transformer (DistilBERT) | 0.5167 | 0.3919 | 0.6667 | 0.0 | 0.7692 | 0.5238 | 0.0 | 60 |

*Evaluation performed with stratified 70/15/15 split without data leakage.*
