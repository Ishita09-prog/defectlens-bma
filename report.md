# Short Report — Bayesian Model Averaging for Software Defect Prediction

**Problem.** Identify defect-prone modules while quantifying model uncertainty. **Data.** NASA MDP/PROMISE (CM1, JM1, KC1, KC3, MC2, MW1, PC1, PC3, PC4); 21–37 static code metrics; 8–35 % defective (after removing exact duplicates).

**Method.** Preprocessing (signed log1p, drop constant and |ρ|>0.95 metrics, z-score; fitted on train only). Candidate models: Bayesian logistic regressions over metric subsets (greedy BIC forward selection + 500-step MC3 search; top 60 kept) and Gaussian Naive Bayes. Posterior model weights P(Mk|D) ∝ exp(−τ·BIC_k/2). Each logistic member has a Laplace posterior over coefficients; predictions are averaged over models and parameter draws, giving credible intervals, aleatoric/epistemic decomposition and posterior inclusion probabilities.

**Protocol.** 60/20/20 stratified split, identical test set for all algorithms; tuning on validation only; repeated over 10 splits (PC4) and 5 splits × 9 datasets.

**Results (mean test, 9 datasets × 5 splits).** ROC-AUC: LogReg 0.770, BMA 0.753, RF 0.753, best-BIC 0.742, NB 0.724. Brier: LogReg 0.117, RF 0.121, BMA 0.120, NB 0.252. ECE: BMA 0.066 ≈ LogReg 0.067 < RF 0.072 ≪ NB 0.262. On PC4 (10 splits): BMA 0.913 ± 0.018 AUC, Brier 0.077.

**Findings.** BMA is competitive and well calibrated, outperforms any single BIC-selected model and Naive Bayes, but is not uniformly the most accurate (LogReg is marginally better on average). Its added value is the predictive distribution: ranking modules by epistemic uncertainty and deferring the most uncertain to review improves AUC/Brier on the remainder (selective-prediction plot).

**Limitations.** BIC weights concentrate on few models for large n (JM1); Laplace posteriors are approximate; NASA MDP lacks timestamps so a random split may be optimistic; Naive Bayes is miscalibrated due to correlated metrics.
