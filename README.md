# Anomaly Detection in Smart Systems Using LSTM-Autoencoder and CTGAN Augmentation

A thesis project that exposes a fundamental evaluation flaw in IoT/smart-system anomaly detection benchmarks and proposes a fix using generative data augmentation.

---

## Research Problem

Standard smart-system anomaly datasets simulate attacks so weakly that the "anomalies" are statistically indistinguishable from normal traffic (Cohen's d < 0.08 across all features). This means anomaly detection models evaluated on such datasets appear to fail — not because the models are poor, but because the benchmark itself is broken.

This project:
1. **Quantifies** the weak simulation problem using Cohen's d effect sizes
2. **Fixes** it using CTGAN to generate realistic attack traffic (Cohen's d 0.17–0.87)
3. **Proposes LSTM-Autoencoder** as the detection model and benchmarks it against baselines
4. **Explains** model decisions using SHAP with a surrogate Random Forest

---

## Key Results

### Detection Performance — Original vs CTGAN-Augmented Evaluation

| Model | Orig F1 | Orig AUC-ROC | CTGAN F1 | CTGAN AUC-ROC |
|---|---|---|---|---|
| Isolation Forest | 0.1067 | 0.5069 | 0.3862 | 0.8073 |
| One-Class SVM | 0.1201 | 0.5124 | 0.7258 | 0.8951 |
| Autoencoder | 0.1231 | 0.5144 | 0.8321 | 0.9463 |
| **LSTM-AE (proposed)** | **0.1282** | **0.5063** | **0.9488** | **0.9978** |

> Original AUC ~0.50 (no better than random) is caused by the weak dataset, not model failure. Under realistic CTGAN evaluation, LSTM-AE achieves AUC = 0.9978.

### XAI Quality (SHAP Surrogate)

| Test Set | Fidelity | Stability |
|---|---|---|
| Original (weak simulation) | 0.9371 | 0.9816 |
| CTGAN (realistic) | 0.9025 | 0.9856 |

High fidelity and stability on both sets confirms the SHAP explanations are trustworthy.

---

## Project Structure

```
thesis_project/
├── data/
│   └── smart_system_anomaly_dataset__1_.csv   # Raw dataset
├── notebooks/
│   ├── 01_preprocessing.ipynb                 # EDA, feature engineering, train/test split
│   ├── 02_isolation_ocsvm.ipynb               # Baseline: Isolation Forest + One-Class SVM
│   ├── 03_autoencoder.ipynb                   # Baseline: Dense Autoencoder
│   ├── 04_lstm_ae.ipynb                       # Proposed: LSTM Autoencoder
│   ├── 05_ctgan.ipynb                         # CTGAN augmentation + central results table
│   └── 06_shap.ipynb                          # SHAP explainability (fidelity + stability)
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── statistical_analysis.csv
│   ├── iso_confusion.png
│   ├── ocsvm_confusion.png
│   ├── ae_loss.png
│   ├── ae_error_distribution.png
│   ├── lstmae_loss.png
│   ├── shap_importance.png
│   └── shap_beeswarm.png
└── preprocessed_data.pkl                      # Shared state passed between notebooks
```

---

## Setup & Requirements

**Python 3.10+**

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy ctgan shap jupyter
```

---

## How to Run

Run notebooks **in order** — each one saves state to `preprocessed_data.pkl` for the next.

```bash
jupyter notebook
```

Then open and run all cells in sequence:

```
01_preprocessing.ipynb  →  02_isolation_ocsvm.ipynb  →  03_autoencoder.ipynb
→  04_lstm_ae.ipynb  →  05_ctgan.ipynb  →  06_shap.ipynb
```

Or execute from the command line:

```bash
for nb in notebooks/0*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

> Note: Notebook 05 (CTGAN training, 150 epochs) takes approximately 5–10 minutes.

---

## Dataset

**Smart System Anomaly Dataset** — tabular IoT/smart-system traffic records with the following features:

| Feature | Type |
|---|---|
| cpu_usage, memory_usage | Resource utilization |
| network_in_kb, network_out_kb, packet_rate | Network traffic |
| avg_response_time_ms | Latency |
| service_access_count, failed_auth_attempts | Access behaviour |
| is_encrypted, geo_location_variation | Security indicators |
| device_type | Categorical |

Engineered features added: `auth_rate`, `network_asymmetry`, `traffic_density`, `cpu_mem_stress`, `response_per_packet`

Label distribution: Normal + 3 anomaly classes (total 16 features after engineering).

---

## Model Architecture

### LSTM Autoencoder (Proposed)
- Input reshaped into sequences: `(batch, seq_len=4, step_dim=4)`
- Encoder: 2-layer LSTM → linear bottleneck (latent dim = 16)
- Decoder: linear projection → 2-layer LSTM → output layer
- Trained on **normal traffic only** (unsupervised)
- Anomaly score = mean per-step reconstruction error
- Threshold = 95th percentile of training reconstruction errors

### Baselines
- **Isolation Forest** — 300 trees, contamination = 0.05
- **One-Class SVM** — RBF kernel, nu = 0.05
- **Dense Autoencoder** — 3-layer encoder (64→32→8) + symmetric decoder

---

## Novel Contributions

1. **Benchmark evaluation flaw** — demonstrated that Cohen's d < 0.08 in the original dataset renders all anomaly scores near-random (AUC ≈ 0.50)
2. **CTGAN augmentation** — generative augmentation raises Cohen's d to 0.17–0.87, creating a fair evaluation benchmark
3. **LSTM-AE superiority** — proposed model achieves F1 = 0.9488 and AUC = 0.9978 under realistic evaluation, outperforming all baselines
4. **XAI validation** — SHAP surrogate achieves fidelity > 0.90 and stability > 0.98 on both original and CTGAN test sets
