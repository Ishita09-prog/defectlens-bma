# DefectLens — AI-Based Software Defect Prediction with Bayesian Model Averaging

Identify defect-prone software modules **while quantifying model uncertainty** (NASA / PROMISE MDP datasets).

| | |
|---|---|
| Primary algorithm | Bayesian Model Averaging (Bayesian logistic regressions over metric subsets via BIC + MC3, Laplace parameter posteriors, Gaussian NB) |
| Comparison | Logistic Regression · Random Forest · Naive Bayes (+ best-single-BIC ablation) |
| Metrics | ROC-AUC, F1, Brier, log-loss (+ PR-AUC, ECE) |
| Outputs | defect-risk dashboard, posterior model weights, uncertainty plots, calibration, comparison table, Streamlit demo |

## Run
**Google Colab:** upload `Defect_Prediction_BMA_Colab.ipynb` → *Runtime → Run all*. The last cell prints a public link to the web demo.

**Local**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Datasets are bundled in `data/` (source: github.com/klainfo/NASADefectDataset, cleaned MDP D'); `bma.load_dataset` re-downloads if missing.

## Protocol
Exact duplicates removed → stratified 60/20/20 split → preprocessing fitted on train only → hyper-parameters & F1 threshold tuned on validation only → **same held-out test set for every algorithm** → repeated over 10 seeds and 9 datasets.

## Files
`bma.py` core library · `app.py` glassmorphism Streamlit app · `data/` · `report.md` short report · notebook.
