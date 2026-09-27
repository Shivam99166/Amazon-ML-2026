# Amazon ML Challenge — Business Entity Resolution

## 1. Project Purpose
This project is built for the **Amazon ML Challenge: Business Entity Resolution**. 
In commercial platforms, business identity data arrives from multiple independent sources with noisy and inconsistent fields (abbreviations, punctuation, missing components, transliterations). 

The goal is to determine which records across independent sources refer to the same real-world business entity:
- **Source 1 (`S1-`)** is the deduplicated reference source.
- Find all matching entity records from **Source 2 (`S2-`)** and **Source 3 (`S3-`)** for each Source 1 entity.
- An entity in Source 1 may match zero (singleton), one, or multiple records across Sources 2 and 3.
- Submissions are evaluated using a macro-averaged **$F_{0.5}$ score** (precision-heavy metric).

---

## 2. Repository Structure
```text
Amazon ml/
├── .gitignore               # Root gitignore excluding datasets, virtualenvs, and outputs
├── README.md                # Project documentation and team collaboration guide
├── requirements.txt         # Pinned Python package dependencies
├── requirment.txt           # Legacy empty file (retained for backward compatibility)
├── data/
│   ├── train/               # Local training files (ignored by Git)
│   │   ├── .gitkeep
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/                # Local test files (ignored by Git)
│       ├── .gitkeep
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── models/                  # Saved model artifacts (ignored by Git)
├── notebooks/               # Jupyter notebooks for experiments & EDA
├── output/                  # Generated predictions & candidate pairs (ignored by Git)
└── src/                     # Core pipeline source code
    ├── 01_inspect.py        # Dataset inspection & verification
    ├── 01_prepare.py        # Data cleaning & normalization
    ├── 03_blocking.py       # Candidate pair generation / blocking
    ├── 04_features.py       # Pairwise feature extraction
    ├── 05_train.py          # Model training & hyperparameter tuning
    ├── 06_validate.py       # Validation & macro F_0.5 threshold optimization
    ├── 07_predict.py        # Test set inference
    └── 08_submission.py     # Output packaging & validation
```

---

## 3. Python Environment Setup

Clone the repository and set up a clean Python virtual environment.

### Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Dataset Setup

> **IMPORTANT**: The large TSV dataset files (~1.8 GB compressed, ~12.5M rows total) are **NOT stored in GitHub** to keep the repository lightweight and performant.

Each teammate must obtain the official dataset and place the files locally in their respective folders:

- **Training data** goes into `data/train/`:
  - `train_source1.tsv`
  - `train_source2.tsv`
  - `train_source3.tsv`
  - `train_ground_truth.tsv`
- **Test data** goes into `data/test/`:
  - `test_source1.tsv` (or `test_source1 (2).tsv`)
  - `test_source2.tsv` (or `test_source2 (2).tsv`)
  - `test_source3.tsv` (or `test_source3 (2).tsv`)

The `.gitignore` is preconfigured to prevent these data files from being committed or pushed.

---

## 5. Pipeline Architecture

The end-to-end entity resolution workflow consists of the following modular steps:

```
Raw Data (TSVs)
       ↓
Data Preparation (Cleaning & Text Normalization)
       ↓
Candidate Blocking (Reducing 17.2T pairs to top-k candidates)
       ↓
Feature Engineering (String, Address & Phonetic Similarities)
       ↓
Model Training (Histogram Gradient Boosting / LightGBM)
       ↓
Validation (Macro F_0.5 Metric & Optimal Threshold Search)
       ↓
Prediction (Inference on Unlabeled Test Pool)
       ↓
Submission (Format Check & Package Generation)
```

---

## 6. Team Git Workflow

To ensure smooth collaboration among 3 teammates without merge conflicts:

1. **`main` Branch**:
   - The `main` branch is protected and always represents the stable, integrated pipeline.
   - Do **NOT** commit directly to `main`.
2. **Feature Branches**:
   - Each teammate creates a dedicated branch for their assigned task (e.g., `feature/blocking`, `feature/features`, `feature/modeling`).
3. **Commit & Push**:
   - Commit descriptive, incremental changes.
   - Push your feature branch to GitHub.
4. **Pull Requests (PR)**:
   - Open a Pull Request from your feature branch to `main`.
   - Have at least one teammate review the PR before merging.
5. **Collaborator Permissions**:
   - The repository owner must grant **Collaborator / Write access** to all 3 teammates in GitHub repo settings (`Settings > Collaborators > Add people`).

---

## 7. Example Workflow for Teammates

```bash
# 1. Clone the repository
git clone <REPOSITORY_URL>
cd "Amazon ml"

# 2. Setup virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Create and switch to your feature branch
git checkout -b feature/blocking

# 4. Make your code modifications, stage, and commit
git add .
git commit -m "Implement candidate blocking"

# 5. Push your branch to GitHub
git push origin feature/blocking

# 6. Open a Pull Request on GitHub and merge once reviewed
```

---

## 8. Important Team Guidelines
- Never remove `.gitignore` rules for datasets, `output/`, or `.venv/`.
- Ensure all submission files conform to the exact format (`output/matching_results.tsv` and `output/candidate_pairs.tsv`).
- Test scripts using local subsets before running full 12M-row pipelines.
