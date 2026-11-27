# Smart Hostel: Autonomous Facility Complaint Classification, Priority Triage & Lifecycle Tracking System

---

## 📑 Table of Contents
1. [System Overview](#system-overview)
2. [Data Engineering & Preprocessing](#data-engineering)
3. [Model Architectures](#model-architectures)
4. [Benchmark Results (Sentiment)](#benchmark-results)
5. [Priority Formulation](#priority-formulation)
6. [Routing & Safeguards](#routing)
7. [Streamlit UI](#streamlit-ui)
8. [Installation & Quick Start](#installation)
9. [Project Structure](#structure)
10. [Authors](#authors)

---

## 🏛️ System Overview

The platform automates the end‑to‑end lifecycle of facility‑maintenance complaints in large residential complexes:
- Residents submit a complaint (text, location, urgency, optional photo).
- The backend sanitises the input, runs a **sentiment** classifier and (optional) a **category** classifier, then computes a transparent priority score.
- High‑confidence predictions are dispatched automatically to the appropriate crew; low‑confidence cases are routed to a *Warden Review* queue.
- An interactive Streamlit dashboard visualises tickets, priorities, crew assignments and analytics.

---

## 📊 Data Engineering & Preprocessing

The raw dataset consists of the **NYC HPD 311 Housing Maintenance Complaints** (≈210 k rows). Key preprocessing steps (implemented in `src/data/`):
1. **Attribute Extraction** – combine `complaint_type` and `descriptor` into a clean `description` field.
2. **Sentiment Mapping** – map the `status` column to sentiment (`negative`, `neutral`, `positive`).
3. **PII Redaction** – strip phone numbers, e‑mail addresses, student IDs.
4. **Deduplication** – drop duplicate descriptions, leaving **123 unique records**.
5. **Stratified Partitioning** – create leakage‑free splits:
   - **Training**: 768 records (original + augmentations)
   - **Validation**: 10 records
   - **Test**: 22 records (held‑out)
   - **External Hold‑out**: 19 records for final evaluation

All splits are stored under `data/processed/`.

---

## 🧠 Model Architectures (Sentiment Only)

Three independent sentiment models were trained on the same training split:
- **TF‑IDF + Logistic Regression** – lightweight linear baseline.
- **Bidirectional LSTM** – 2‑layer BiLSTM (`hidden=128`, dropout=0.4) built with PyTorch.
- **Fine‑Tuned DistilBERT** – Hugging‑Face `distilbert-base-uncased` fine‑tuned for sequence classification.

The category models remain in the codebase for completeness but are omitted from the benchmark presentation.

---

## 📈 Benchmark Results (Sentiment)

All models were evaluated on the **22‑record held‑out test set**. Only test accuracy is shown for simplicity:

| Model | Test Accuracy |
| :--- | :---: |
| **TF‑IDF + Logistic Regression** | **63.6 %** |
| **Bidirectional LSTM (BiLSTM)** | **63.6 %** |
| **Fine‑Tuned DistilBERT** | **77.3 %** |

These numbers match the entries in `results/benchmark_summary.json`.

---

## 🧮 Priority Formulation

The final priority score `P` (0 – 100) is a weighted sum of four factors:

```
P = 0.35·C + 0.30·U + 0.20·R + 0.15·I
```
- **C** – baseline risk for the complaint category (e.g., electrical = 85).
- **U** – urgency cues extracted from the text (hazard keywords → high scores).
- **R** – resident‑selected urgency level (Emergency = 100, Low = 25).
- **I** – impact scope (whole wing = 100, single room = 25).

Safety overrides ensure any high‑risk keyword (e.g., *fire*, *spark*) bumps `P` to ≥ 85 regardless of other factors.

---

## 🚦 Routing & Human‑in‑the‑Loop Safeguards

- **Crew Mapping** – each category maps to a dedicated maintenance crew.
- **Confidence Gating** – predictions with confidence `< 0.60` are sent to the *Warden Review* queue.
- **Sentiment Separation** – sentiment is logged purely for resident satisfaction tracking and never influences safety‑critical routing.

---

## 🖥️ Streamlit UI

The dashboard (`src/app.py`) offers four panels:
1. **Resident Portal** – submit complaints and view instant triage results.
2. **Staff Dispatch** – filter tickets by crew, track progress.
3. **Warden Analytics** – view KPIs, hotspot maps and review low‑confidence tickets.
4. **Benchmark Hub** – display the simplified sentiment benchmark table above.

---

## 🚀 Installation & Quick Start

```bash
# Clone the repo
git clone https://github.com/harshitchauhann95/Project-472.git
cd Project-472

# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Prepare data (uses the full 210k raw file)
python src/data/download_dataset.py          # pulls the raw CSV
python src/data/preprocess.py               # cleaning & sentiment mapping
python src/data/split_data.py               # creates the 768/10/22 splits

# Train and evaluate sentiment models
python src/models/train_benchmarks.py
python src/evaluate.py

# Launch the UI
streamlit run src/app.py
```

---

## 📁 Project Structure

```
Project-472/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # original NYC HPD CSV (git‑ignored)
│   └── processed/          # train.csv, val.csv, test.csv, external_holdout.csv
├── models/                 # saved model weights & tokenizers
├── results/                # benchmark_summary.json, plots
└── src/
    ├── data/               # download_dataset.py, preprocess.py, split_data.py
    ├── models/             # tfidf_baseline.py, bilstm_model.py, transformer_model.py
    ├── routing/            # priority_engine.py, team_router.py
    ├── evaluate.py
    └── app.py
```

---

## 👥 Authors
- **Harshit Chauhan** – Lead architecture & implementation
- **Vishnu Kumar Jha**
- **Alok Kumar**

*Department of Computer Science & Engineering*
