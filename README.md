# Noise-Robust, Leakage-Safe Transaction Categorization for Indian Banking

**Live demo:** https://ai-personal-finance-advisor-lazazyvemk4h9ghxwawblo.streamlit.app/

**Source code:** https://github.com/vishvacodes/AI-Personal-Finance-Advisor
An evidence-based, privacy-first, and grounded **AI Personal Finance Advisor** built with Python, Scikit-Learn, LightGBM, XGBoost, pandas, and Streamlit.

> **Core Thesis**:
> *"Noise-robust, leakage-safe, personalizable transaction categorization for Indian bank statements, with a grounded advisor."*
> The machine learning benchmarks serve as supporting evidence for this thesis, not the headline.

---

## 1. System Architecture

```
                       RAW INDIAN BANK STATEMENT
                                   │
                                   ▼
                  DOMAIN CANONICALIZER (src/canonicalize.py)
   (Strips UPI VPAs, POS slips, NEFT prefixes, IMPS IDs, masked cards)
                                   │
                                   ▼
                   TIERED PERSONALIZED CLASSIFIER
   ┌───────────────────────────────────────────────────────────────┐
   │ Level 1: Global Calibrated Linear SVM (Fallback baseline)     │
   │ Level 2: User Correction Memory (Exact / Normalized match)    │
   │ Level 3: Adaptive Cosine k-NN Layer (Active at N >= 5 edits)   │
   └───────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                      VERIFIED TRANSACTION LABELS
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
         BEHAVIORAL ENGINE              DETERMINISTIC ANALYTICS
    (Cash Flow & Discretionary)    (Budgeting, Goals, Forecasting)
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
               OUTBOUND PRIVACY FIREWALL (src/privacy.py)
      (38-Field Allowlist + Indian PII Redaction: PAN, Aadhaar, Phone)
                                   │
                                   ▼
               DETERMINISTIC PYTHON FINANCIAL TOOLS
         (Spending aggregations, anomaly detection, forecasts)
                                   │
                                   ▼
            NUMERIC GROUNDING VALIDATOR (src/numeric_validator.py)
    (Asserts every number, %, and currency matches tool-computed facts)
                                   │
                                   ▼
          GROUNDED FINANCIAL ADVISOR (src/grounded_llm.py)
      (Provider-agnostic: Ollama / OpenAI-compatible / Offline Fallback)
```

---

## 2. Key Architectural Contributions

### 1. Adaptive NLP & Domain Canonicalization (`src/canonicalize.py`, `src/adaptive_nlp.py`)
- **Domain Canonicalization**: Standardizes Indian banking transaction noise patterns (`UPI/123/SWIGGY/vpa@okaxis` $\rightarrow$ `SWIGGY`, `POS 4129XXXX9812 RELIANCE MUMBAI` $\rightarrow$ `RELIANCE`, `NEFT-928-RENT` $\rightarrow$ `RENT`). Boosts accuracy by **+22.20%** on held-out IMPS and **+12.98%** on held-out NEFT without model retraining.
- **Multi-Tier Personalization**: Combines the global classifier with user-specific correction memory and prototype k-NN. Increases user accuracy from **38.85%** ($N=0$) to **88.89%** ($N=20$) while maintaining strict multi-tenant isolation.

### 2. Behavioral Modeling & Discretionary Tracking (`src/behavioral_engine.py`)
- Computes individual cash flow ratios, recurring transaction cadence, and discretionary spending distributions (Dining, Entertainment, Shopping).

### 3. Personalized Forecasting & Backtesting (`src/forecast.py`)
- 1-step-ahead spending forecasting with 80% empirical uncertainty intervals.
- Validated via rolling-origin backtesting across multi-month financial profiles.

### 4. Explainable AI & Statistical Calibration (`src/explainability.py`, `src/calibration.py`)
- N-gram feature weights for every prediction.
- Post-hoc temperature scaling ($T=0.6453$) fitted on noisy validation data, reducing Expected Calibration Error (ECE) from 0.1459 to 0.1321.
- Confidence-based risk-coverage routing to flag uncertain transactions for human review.

### 5. Grounded Advisor & Numeric Hallucination Defense (`src/numeric_validator.py`, `src/grounded_llm.py`)
- **Strict Grounding**: Deterministic Python functions compute financial numbers; the LLM merely translates verified data into natural language.
- **Numeric Grounding Validator**: Automatically extracts every scalar, currency amount, and percentage in the LLM response and ensures it exists in tool data. Caught **100% of injected numerical hallucinations** (25/25) during evaluation, falling back cleanly to deterministic templates.
- **Privacy Firewall (`PrivacyGuard`)**: Restricts outbound payloads to a 38-field allowlist and automatically redacts Indian PAN, Aadhaar, credit card, and mobile phone numbers.

---

## 3. Verified Multi-Model Benchmark Results

Evaluated on identical zero-leakage partitions from [`results/final_metrics.csv`](results/final_metrics.csv) and [`results/prf_summary.csv`](results/prf_summary.csv) (generated by `scripts/step2_prf.py`). N=20,040 held-out test split; N=105 real Indian bank statements.

### 3.1 Accuracy & Macro-F1 Summary

| Model | Clean Acc | Clean Macro-F1 | Noisy V2 Acc | Noisy V2 Macro-F1 | Char-swap (N=3,985) Acc | Indian N=105 Acc (11-class) | Indian N=100 Acc (10-class) | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | **0.9913** | **0.9913** | 0.6900 | 0.6960 | 0.7752 | **0.9333** | **0.9800** | GPU batch |
| **Candidate SVM** (`variant_b_candidate.joblib`) | 0.9900 | 0.9900 | **0.7639** | **0.7660** | **0.8806** | 0.8762 | 0.9200 | 4.96 ms |
| **LightGBM (20k)** (`lightgbm_transaction_classifier.joblib`) | 0.9772 | 0.9772 | 0.6355 | 0.6408 | 0.7930 | 0.7810 | 0.8200 | **1.97 ms** |
| **MiniLM-L6-v2 + LogReg** | 0.8967 | 0.8969 | 0.4781 | 0.4898 | 0.5694 | 0.8095 | 0.8500 | 9.73 ms |
| **XGBoost** | 0.7261 | 0.7318 | 0.4514 | 0.4623 | 0.5601 | 0.7048 | 0.7400 | 1.55 ms |
| **Legacy Baseline** (`final_model.joblib`) | 0.9907 | 0.9907 | 0.6389 | 0.7014 | 0.8968 | 0.5619 | 0.5900 | 4.76 ms |

*All numbers from `results/prf_summary.csv`, `results/n96_per_row_predictions.csv`, and `results/distilbert_preds_indian_105.csv`. Indian N=105 (11-class) includes Rent (5 rows); recall=0.00 on Rent for all models — Rent is not a trained class. Indian N=100 (10-class) is computed from saved per-row predictions with Rent rows excluded; the 10-class correct count equals the 11-class correct count (no Rent rows were predicted correctly), but the denominator drops from 105 to 100. Char-swap subset = `noise_type==char_swap` (N=3,985). LightGBM (20k) = `lightgbm_transaction_classifier.joblib` (20,000 char-wb features, deployed app model).*

*The rent rule is an offline post-classification heuristic used in evaluation scripts only — not deployed in the Streamlit app. Model-only figures are the headline numbers.*

### 3.2 Weakest Classes (Indian N=105, model-only)

All models: **Rent recall = 0.00** — confirmed from saved per-row predictions (`results/n96_per_row_predictions.csv`). Rent is not a trained class; all 5 Rent rows are misclassified (SVM: Government & Legal, Income, Charity & Donations, Income, Healthcare & Medical).  
Beyond Rent, the next weakest classes:

- **Candidate SVM**: Healthcare & Medical (0.80), Shopping & Retail (0.87), Transportation (0.87)
- **LightGBM**: Transportation (0.53), Food & Dining (0.73)
- **DistilBERT**: Food & Dining (0.93), Shopping & Retail (0.93) — all other classes ≥0.93

Top confusion: Rent → Income (DistilBERT: 4/5, SVM: 2 Income + 1 Gov + 1 Charity + 1 Healthcare). See `results/confusion_matrix_*.png`.

### 3.3 Architectural Findings by Data Type

- **Linear SVM**: Character n-grams are naturally linearly separable. Candidate SVM achieves the highest robustness on independent noise (Noisy V2 macro-F1: 0.7660), significantly outperforming DistilBERT (0.6960). DistilBERT leads on clean text (0.9913 vs 0.9900).
- **LightGBM**: The deployed app uses `lightgbm_transaction_classifier.joblib` (20k-equivalent features) — 0.9772 clean macro-F1, 1.97 ms latency. The matched 3k ablation reference (`matched_lightgbm_3k.joblib`) reaches 0.9732 clean macro-F1 / 0.6567 Noisy V2 / 0.8095 Indian N=105 (see `results/lightgbm_3k_vs_20k.csv`).
- **XGBoost gap**: The original `xgboost_transaction_classifier.joblib` scored 0.7261 clean accuracy (weaker config). Matching the 3k feature setup (variant a) recovers **+16.2 pp → 0.8882**. Further: +1.3 pp dense (b), +1.6 pp lossguide (c), combined +2.65 pp (d), +2.0 pp tuning (e) → 0.9351. Accuracy gaps to LightGBM 3k: clean **−3.8 pp**, Noisy V2 **−10.9 pp**, Indian N=105 **−4.8 pp**. Cause unproven; EFB untested — see `results/xgb_ablation/xgb_ablation_results.csv`.
- **MiniLM**: Captures semantic nuance (Indian N=105 macro-F1: 0.7606) but degrades severely on character noise (Noisy V2: 0.4898).
- **DistilBERT**: Peak clean-text performance; drops to 0.6960 macro-F1 under independent noise (ECE=0.24, severely overconfident). Fine-tuned on original seed — Indian results are preliminary.

**Deployed model**: Candidate Linear SVM (`variant_b_candidate.joblib`) — chosen for noise robustness and low single-call latency (4.96 ms).

### 3.4 Canonicalization Gain Per Noise Type

| Noise Corruption | Raw Input Acc | Canonicalized Acc | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **IMPS (Held-out from training)** | 67.24% | **89.44%** | **+22.20%** |
| **NEFT (Held-out from training)** | 76.59% | **89.57%** | **+12.98%** |
| **UPI** | 79.52% | **89.41%** | **+9.89%** |
| **POS** | 80.52% | **89.31%** | **+8.79%** |
| **UPPERCASE** | 80.89% | **88.94%** | **+8.05%** |
| **TRUNCATION** | 80.99% | **88.59%** | **+7.60%** |
| **TYPO / PERMUTATION** | 88.58% | **89.18%** | **+0.60%** |

---

## 4. Statistical Honesty & Integrity Audit

1. **Real Indian Test Set ($N=105$, `data/own_test_indian.csv`)**:
   - Primary Model-Only accuracies: DistilBERT: **93.33%** ($N=105$) / **92.71%** ($N=96$) / **87.93%** ($N=58$); Candidate Linear SVM: **87.62%** ($N=105$) / **86.46%** ($N=96$) / **81.03%** ($N=58$); MiniLM-LogReg: **80.95%** ($N=105$); LightGBM (3k-matched artifact `matched_lightgbm_3k.joblib`): **78.10%** ($N=105$) / **80.21%** ($N=96$) / **74.14%** ($N=58$); LightGBM (20k artifact `lightgbm_transaction_classifier.joblib`, used in app): **77.08%** ($N=96$); XGBoost: **70.48%** ($N=105$). Rule-assisted (+Rent) results are secondary.
   - McNemar statistical tests reveal that differences between top models on the $N=105$ sample have $p$-values between 0.12 and 0.65 ($p > 0.05$). On the N=96 non-overlapping subset, DistilBERT vs Candidate SVM: exact p=0.03125 (significant at α=0.05; not significant after Bonferroni correction; treat as suggestive).
2. **Synthetic Training Seed Provenance & Leakage Removal**:
   - `data/indian_merchants_train_seed.csv` (3,062 rows) was generated using templates of known Indian merchants.
   - 8 merchants overlapped between the training seed and the real test set; these were removed and retested for SVM, LightGBM, XGBoost, and MiniLM. DistilBERT was NOT retrained on the leakage-free seed, so its Indian results are preliminary.
3. **Synthetic Noise Function Sharing**:
   - The initial noisy test set shared noise code with training and is described only as "initial, optimistic" (e.g. Linear SVM 97.19%).
   - An independent noise generator was built for **Noisy V2** ($N=20,040$), which serves as the headline noisy benchmark (Linear SVM: 76.39%, DistilBERT: 69.00%, LightGBM: 63.55%).

---

## 5. Limitations

1. **Rent class absent from training**: No ML model was trained on Rent. Recall=0.00 for all models on the 5 Rent rows in the Indian test set. The rent rule (offline post-classification heuristic in evaluation scripts only) is not deployed in the app.
2. **Small real Indian test set**: The real Indian test set is small ($N=105$); performance differences between top models are not statistically significant. Four classes have only 5 rows each.
3. **Seed-test merchant overlap**: 8 merchants overlapped between the training seed and the real test set; these were removed and retested for SVM, LightGBM, XGBoost, and MiniLM. DistilBERT was NOT retrained on the leakage-free seed, so its Indian results are preliminary.
4. **Shared noise generator**: The first noisy test set shared noise code with training; an independent noisy test set (Noisy V2) was built to evaluate out-of-distribution robustness.
5. **Synthetic benchmark nature**: The Mitul Shah benchmark (100k transactions) is likely synthetic; clean accuracy near 99% reflects an easy, well-separated task.
6. **Single run variance**: Benchmarks use a single seed and a single run for all models; multi-seed variance was not evaluated across all architectures.
7. **Transformer calibration under noise**: DistilBERT is severely overconfident under character noise (ECE=0.24 on Noisy V2).
8. **Rent rule origin and scope**: The rent rule is an offline post-classification heuristic used in evaluation scripts only. It is not applied in the deployed Streamlit app, whose rent transactions may be misclassified (e.g. 'MONTHLY APARTMENT RENT TRANSFER TO LANDLORD' → Government & Legal). Its keywords match the Rent rows of the test set, so +rent figures are not evidence of generalization. Model-only figures are the headline numbers.
9. **Ambiguous grocery labeling**: Quick-commerce grocery apps (BigBasket, Blinkit, Zepto) exhibit ambiguous ground truth labels across Food & Dining vs Shopping & Retail.
10. **Text-only inputs**: The classification engine is text-only and does not incorporate transaction amount, timestamp, or Merchant Category Codes (MCC).
11. **XGBoost gap not fully ablated**: The original model scored 0.7261 clean accuracy (weaker config). Variant a (same 3k features) recovers +16.2 pp → 0.8882; tuning (variant e) reaches 0.9351, leaving accuracy gaps to LightGBM 3k: clean −3.8 pp, Noisy V2 −10.9 pp, Indian N=105 −4.8 pp. EFB is an untested hypothesis.

---

## 6. Getting Started & Reproducibility

### 6.1 Installation
Clone the repository and install requirements in Python 3.10+:
```bash
git clone https://github.com/dhpcodes/AI-Personal-Finance-Advisor.git
cd AI-Personal-Finance-Advisor
pip install -r requirements.txt
```

### 6.2 Running the Full Test Suite
Execute all 106 unit, integration, privacy, and regression tests:
```bash
pytest -o pythonpath=.
```
Expected output:
```
106 passed, 1 skipped in ~70s
```

### 6.3 Reproducing Benchmarks & Evaluations
```bash
# 1. Noise robustness ablation matrix & per-noise-type evaluation
python scripts/run_phase2_noise_robustness.py

# 2. Boosted tree tuning post-mortem (Optuna + Stacking)
python scripts/run_phase3_boosted_trees.py

# 3. Personalization simulation across N feedback steps
python scripts/run_phase5_personalization.py

# 4. Grounding validator & hallucination benchmark
python scripts/run_phase6_grounding_eval.py

# 5. Statistical honesty (Wilson CIs & multi-seed evaluation)
python scripts/run_phase7_seeds_and_stats.py
```

### 6.4 Running the Application
#### Option A: Pure Offline Mode (No LLM, No Internet)
The core financial advisor operates completely locally without an LLM or internet connection:
```bash
streamlit run app/app.py
```

#### Option B: Local LLM Mode via Ollama
Ensure [Ollama](https://ollama.ai) is running locally with Llama 3.2:
```bash
ollama run llama3.2:3b
```
Configure `.env`:
```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2:3b
```
Then launch the Streamlit app:
```bash
streamlit run app/app.py
```

---

## 7. Repository Structure

```
├── app/
│   └── app.py                     # Interactive Streamlit application
├── data/
│   ├── mitulshah_split.csv        # Deduplicated unseen merchant split
│   └── own_test_indian.csv        # Held-out real Indian bank statements (N=105)
├── models/
│   ├── final_model.joblib              # Production Baseline (legacy, read-only)
│   └── variant_b_candidate.joblib     # Candidate Linear SVM (20k features, app default)
├── notebooks/                     # Colab workflows for GPU models (ModernBERT, QLoRA)
├── results/                       # Measured metrics, ablation CSVs, and figures
│   ├── EXPERIMENT_REPORT.md       # Comprehensive experimental findings & audit
│   ├── model_comparison_consolidated.csv
│   ├── noise_ablation_matrix.csv
│   ├── per_noise_type_evaluation.csv
│   └── grounding_evaluation_report.csv
├── scripts/                       # Reproducible phase benchmark scripts
├── src/                           # Production modular architecture
│   ├── adaptive_nlp.py            # Multi-tier personalization layer
│   ├── behavioral_engine.py       # Cash flow & discretionary spending
│   ├── calibration.py             # Post-hoc temperature scaling & ECE
│   ├── canonicalize.py            # Domain-specific banking string normalization
│   ├── database.py                # Multi-tenant SQLite storage & user isolation
│   ├── forecast.py                # Time-series forecasting & backtesting
│   ├── grounded_llm.py            # Provider-agnostic grounded LLM interface
│   ├── numeric_validator.py       # Deterministic numerical hallucination validator
│   ├── predict.py                 # Core transaction classification engine
│   └── privacy.py                 # Outbound allowlist & PII redaction firewall
└── tests/                         # Comprehensive pytest test suite (106 tests)
```

---

## 8. License
MIT License. See [LICENSE](LICENSE) for details.
