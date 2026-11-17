"""
bma.py  -  Bayesian Model Averaging for software defect prediction
===================================================================
Core library used by the notebook and by the Streamlit app.

Pipeline
--------
1. load_dataset()      NASA MDP / PROMISE module-level metrics (local CSV -> GitHub fallback)
2. Preprocessor        signed-log1p -> drop constant -> drop correlated -> standardise
                       (fitted ONLY on training data)
3. BMAClassifier       candidate probabilistic models + BIC posterior model weights +
                       Laplace parameter posteriors -> averaged predictive distribution
4. run_study()         same held-out test set for every algorithm, tuning on validation only
5. repeated_study()    robustness over several random splits
"""
from __future__ import annotations

import io
import os
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

TARGET = "defective"
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

DATASETS = {
    "KC1": "NASA KC1 - C++ storage-management system",
    "PC1": "NASA PC1 - flight software for an earth-orbiting satellite",
    "PC3": "NASA PC3 - flight software for an earth-orbiting satellite",
    "PC4": "NASA PC4 - flight software for an earth-orbiting satellite",
    "CM1": "NASA CM1 - spacecraft instrument in C",
    "JM1": "NASA JM1 - real-time predictive ground system",
    "KC3": "NASA KC3 - storage management for ground data",
    "MW1": "NASA MW1 - zero-gravity experiment (combustion)",
    "MC2": "NASA MC2 - video guidance system",
}
_RAW = ("https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/"
        "CleanedData/MDP/D%27/{n}.arff")


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def _arff_to_df(text: str) -> pd.DataFrame:
    from scipy.io import arff
    data, _ = arff.loadarff(io.StringIO(text))
    df = pd.DataFrame(data)
    t = df.columns[-1]
    s = df[t].map(lambda v: v.decode() if isinstance(v, bytes) else str(v)).str.strip()
    df = df.drop(columns=[t])
    df[TARGET] = s.isin(["Y", "yes", "true", "TRUE", "1"]).astype(int)
    return df


def load_dataset(name: str = "KC1", data_dir: str | None = None) -> pd.DataFrame:
    """Load a NASA MDP dataset: local CSV first, then the public GitHub mirror."""
    name = name.upper()
    for d in filter(None, [data_dir, os.path.join(HERE, "data"), os.path.join(os.getcwd(), "data")]):
        p = os.path.join(d, f"{name}.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    with urllib.request.urlopen(_RAW.format(n=name), timeout=60) as r:
        df = _arff_to_df(r.read().decode("utf-8", "ignore"))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    try:
        df.to_csv(os.path.join(HERE, "data", f"{name}.csv"), index=False)
    except OSError:
        pass
    return df


# ----------------------------------------------------------------------------
# 2. PREPROCESSING  (fit on training data only)
# ----------------------------------------------------------------------------
class Preprocessor:
    """signed log1p -> drop constant cols -> drop |corr|>thr cols -> z-score."""

    def __init__(self, corr_threshold: float = 0.95):
        self.corr_threshold = corr_threshold

    @staticmethod
    def _slog(a):
        return np.sign(a) * np.log1p(np.abs(a))

    def fit(self, X: pd.DataFrame):
        self.raw_columns_ = list(X.columns)
        Z = pd.DataFrame(self._slog(X.values.astype(float)), columns=X.columns)
        keep = [c for c in Z.columns if Z[c].std() > 1e-9]
        corr = Z[keep].corr().abs().fillna(0)
        dropped, final = set(), []
        for c in keep:                       # greedy: keep the first of a correlated pair
            if c in dropped:
                continue
            final.append(c)
            for o in keep:
                if o != c and o not in dropped and o not in final and corr.loc[c, o] > self.corr_threshold:
                    dropped.add(o)
        self.columns_ = final
        self.dropped_constant_ = [c for c in X.columns if c not in keep]
        self.dropped_correlated_ = sorted(dropped)
        self.mean_ = Z[final].mean().values
        self.std_ = Z[final].std().values
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._slog(X[self.columns_].values.astype(float))
        return (Z - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------------
# 3. BAYESIAN MODEL AVERAGING
# ----------------------------------------------------------------------------
def fit_logit(X, y, l2=1.0, max_iter=60):
    """Bayesian logistic regression, MAP via IRLS with N(0, 1/l2) prior.
    Returns (beta, loglik, posterior_covariance[Laplace])."""
    n, d = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])
    pen = np.full(d + 1, float(l2))
    pen[0] = 1e-6
    beta = np.zeros(d + 1)
    for _ in range(max_iter):
        p = expit(np.clip(Xd @ beta, -30, 30))
        w = p * (1 - p) + 1e-9
        g = Xd.T @ (y - p) - pen * beta
        H = (Xd * w[:, None]).T @ Xd + np.diag(pen)
        step = np.linalg.solve(H, g)
        step = np.clip(step, -5, 5)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-7:
            break
    p = expit(np.clip(Xd @ beta, -30, 30))
    w = p * (1 - p) + 1e-9
    H = (Xd * w[:, None]).T @ Xd + np.diag(pen)
    ll = float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    return beta, ll, np.linalg.inv(H)


def _entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class BMAClassifier:
    """Bayesian Model Averaging over probabilistic defect classifiers.

    Candidate models (M_k)
      * Bayesian logistic regressions over many subsets of the code metrics, found by
        greedy BIC forward selection + MC3 (Markov-chain model composition) search
        -> variable-selection uncertainty
      * Gaussian Naive Bayes (all metrics)
      * (optional) shallow decision trees with Laplace-smoothed leaves

    Posterior model weights  P(M_k | D)  ∝  P(M_k) exp(-tau * BIC_k / 2)
    (Schwarz / BIC approximation to the marginal likelihood, Raftery 1995).

    Predictive distribution
      p(y=1|x,D) = sum_k P(M_k|D) * E[p(y=1|x,theta_k,M_k) | D]
    Logistic models carry a Laplace (Gaussian) posterior over coefficients, so every
    prediction also gets a credible interval; uncertainty is decomposed into
      * aleatoric  (expected entropy of the members)
      * epistemic  (mutual information = total entropy - aleatoric)
      * between-model vs within-model variance.
    """

    def __init__(self, max_features=10, n_mc3=500, include_nb=True, include_trees=False,
                 tree_depths=(2, 3, 4), temperature=1.0, prior_precision=1.0, max_models=60,
                 n_draws=2000, random_state=0):
        self.max_features = max_features
        self.n_mc3 = n_mc3
        self.include_nb = include_nb
        self.include_trees = include_trees
        self.tree_depths = tree_depths
        self.temperature = temperature
        self.prior_precision = prior_precision
        self.max_models = max_models
        self.n_draws = n_draws
        self.random_state = random_state

    # ---- fitting -----------------------------------------------------------
    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(p)]
        self.n_train_ = n
        logn = np.log(n)
        cache = {}                                   # subset (sorted tuple) -> fitted logistic model

        def get(S):
            key = tuple(sorted(S))
            if key not in cache:
                beta, ll, cov = fit_logit(X[:, list(key)], y, self.prior_precision, max_iter=25)
                cache[key] = dict(kind="logreg", idx=list(key), beta=beta, cov=cov, loglik=ll,
                                  k=len(key) + 1, bic=-2 * ll + (len(key) + 1) * logn)
            return cache[key]

        # (a1) greedy BIC forward selection -> good starting point
        chosen, remaining = [], list(range(p))
        get([])
        for _ in range(min(self.max_features, p)):
            j = min(remaining, key=lambda j: get(chosen + [j])["bic"])
            chosen.append(j)
            remaining.remove(j)
        # (a2) MC3: Metropolis search over metric subsets  (add / drop / swap moves)
        cur = set(get(chosen)["idx"])
        cur_bic = get(cur)["bic"]
        for _ in range(self.n_mc3):
            moves = []
            if len(cur) < self.max_features:
                moves.append("add")
            if len(cur) > 0:
                moves.append("drop")
            if 0 < len(cur) < p:
                moves.append("swap")
            mv = rng.choice(moves)
            prop = set(cur)
            if mv == "add":
                prop.add(int(rng.choice([j for j in range(p) if j not in cur])))
            elif mv == "drop":
                prop.remove(int(rng.choice(sorted(cur))))
            else:
                prop.remove(int(rng.choice(sorted(cur))))
                prop.add(int(rng.choice([j for j in range(p) if j not in cur])))
            pb = get(prop)["bic"]
            if np.log(rng.random() + 1e-300) < -0.5 * (pb - cur_bic):
                cur, cur_bic = prop, pb
        models = sorted(cache.values(), key=lambda m: m["bic"])[: self.max_models]
        # (b) Gaussian naive Bayes
        if self.include_nb:
            nb = GaussianNB(var_smoothing=1e-3).fit(X, y)
            pr = nb.predict_proba(X)
            ll = float(np.sum(np.log(np.where(y == 1, pr[:, 1], pr[:, 0]) + 1e-12)))
            models.append(dict(kind="nb", est=nb, loglik=ll, k=2 * p + 1, idx=list(range(p)),
                               bic=-2 * ll + (2 * p + 1) * logn))
        # (c) optional shallow trees with Laplace-smoothed leaves
        if self.include_trees:
            for d in self.tree_depths:
                t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=10, random_state=self.random_state).fit(X, y)
                leaves = t.apply(X)
                tab = {lf: (y[leaves == lf].sum() + 1.0) / ((leaves == lf).sum() + 2.0) for lf in np.unique(leaves)}
                pr = np.array([tab[l] for l in leaves])
                ll = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
                models.append(dict(kind="tree", est=t, tab=tab, loglik=ll, k=int(t.get_n_leaves()),
                                   idx=None, bic=-2 * ll + int(t.get_n_leaves()) * logn))
        models.sort(key=lambda m: m["bic"])
        for r, m in enumerate(models, 1):
            m["name"] = f"M{r:02d}-" + {"logreg": "LR", "nb": "NB", "tree": "Tree"}[m["kind"]]
        self.models_ = models
        self.n_visited_ = len(cache)
        self._compute_weights()
        return self

    def _compute_weights(self):
        bic = np.array([m["bic"] for m in self.models_])
        a = -0.5 * self.temperature * (bic - bic.min())
        w = np.exp(a - a.max())
        self.weights_ = w / w.sum()
        return self

    def set_temperature(self, tau: float):
        """tau=1 -> textbook BIC posterior; tau<1 -> flatter (more model uncertainty)."""
        self.temperature = float(tau)
        return self._compute_weights()

    # ---- inspection --------------------------------------------------------
    def weights_table(self) -> pd.DataFrame:
        rows = []
        for m, w in zip(self.models_, self.weights_):
            feats = ""
            if m["kind"] == "logreg":
                feats = ", ".join(self.feature_names_[i] for i in m["idx"])
            rows.append(dict(model=m["name"], kind=m["kind"], params=m["k"],
                             log_likelihood=m["loglik"], BIC=m["bic"], posterior_weight=w,
                             metrics=feats))
        return pd.DataFrame(rows).sort_values("posterior_weight", ascending=False).reset_index(drop=True)

    def inclusion_probabilities(self) -> pd.Series:
        """Posterior inclusion probability of each metric over the logistic candidates."""
        pip = np.zeros(len(self.feature_names_))
        tot = 0.0
        for m, w in zip(self.models_, self.weights_):
            if m["kind"] == "logreg":
                pip[m["idx"]] += w
                tot += w
        pip = pip / tot if tot > 0 else pip
        return pd.Series(pip, index=self.feature_names_).sort_values(ascending=False)

    @property
    def best_model_(self):
        return self.models_[int(np.argmax(self.weights_))]

    # ---- prediction --------------------------------------------------------
    def _member_proba(self, m, X):
        """Posterior-mean-ish prediction of one member (point estimate)."""
        if m["kind"] == "logreg":
            Xd = np.hstack([np.ones((len(X), 1)), X[:, m["idx"]]])
            return expit(Xd @ m["beta"])
        if m["kind"] == "nb":
            return m["est"].predict_proba(X)[:, 1]
        return np.array([m["tab"].get(l, 0.5) for l in m["est"].apply(X)])

    def predict_proba(self, X) -> np.ndarray:
        """BMA predictive probability (fast, no sampling)."""
        X = np.asarray(X, float)
        out = np.zeros(len(X))
        for m, w in zip(self.models_, self.weights_):
            if w > 1e-8:
                out += w * self._member_proba(m, X)
        return out

    def predict_proba_best(self, X) -> np.ndarray:
        return self._member_proba(self.best_model_, np.asarray(X, float))

    def predict_dist(self, X, level=0.90, n_draws=None, seed=None) -> dict:
        """Full predictive distribution with credible intervals and uncertainty split."""
        X = np.asarray(X, float)
        S = int(n_draws or self.n_draws)
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        counts = rng.multinomial(S, self.weights_)
        draws, member_of = [], []
        for mi, (m, c) in enumerate(zip(self.models_, counts)):
            if c == 0:
                continue
            if m["kind"] == "logreg":
                betas = rng.multivariate_normal(m["beta"], m["cov"], size=c, method="cholesky")
                Xd = np.hstack([np.ones((len(X), 1)), X[:, m["idx"]]])
                dr = expit(np.clip(Xd @ betas.T, -30, 30))               # (N, c)
            else:
                dr = np.repeat(self._member_proba(m, X)[:, None], c, axis=1)
            draws.append(dr)
            member_of += [mi] * c
        D = np.hstack(draws)                                           # (N, S)
        member_of = np.array(member_of)
        mean = D.mean(1)
        a = (1 - level) / 2
        lo, hi = np.quantile(D, a, axis=1), np.quantile(D, 1 - a, axis=1)
        total_var = D.var(1)
        mm = np.stack([D[:, member_of == k].mean(1) for k in np.unique(member_of)], 1)
        wk = np.array([(member_of == k).mean() for k in np.unique(member_of)])
        between = (wk * (mm - mean[:, None]) ** 2).sum(1)
        within = np.maximum(total_var - between, 0)
        total_ent = _entropy(mean)
        aleatoric = _entropy(D).mean(1)
        epistemic = np.maximum(total_ent - aleatoric, 0)
        return dict(mean=mean, lo=lo, hi=hi, width=hi - lo, std=np.sqrt(total_var),
                    between_var=between, within_var=within, total_entropy=total_ent,
                    aleatoric=aleatoric, epistemic=epistemic, draws=D, level=level)


# ----------------------------------------------------------------------------
# 4. METRICS
# ----------------------------------------------------------------------------
def ece_score(y, p, n_bins=10):
    """Expected calibration error (equal-width bins)."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability_curve(y, p, n_bins=8):
    """Equal-frequency reliability points: (mean predicted, observed freq, count)."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    chunks = np.array_split(order, n_bins)
    return (np.array([p[c].mean() for c in chunks]), np.array([y[c].mean() for c in chunks]),
            np.array([len(c) for c in chunks]))


def best_f1_threshold(y, p):
    ts = np.unique(np.round(np.quantile(p, np.linspace(0.02, 0.98, 97)), 6))
    f = [f1_score(y, p >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))])


def metrics_at(y, p, thr):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                F1=f1_score(y, p >= thr, zero_division=0),
                Precision=precision_score(y, p >= thr, zero_division=0),
                Recall=recall_score(y, p >= thr, zero_division=0),
                Brier=brier_score_loss(y, p), LogLoss=log_loss(y, p), ECE=ece_score(y, p))


INTERPRETABILITY = {
    "Bayesian Model Averaging": "High - posterior weights, inclusion probabilities, credible intervals",
    "Best single BIC model": "High - one sparse logistic model",
    "Logistic Regression": "High - coefficients / odds ratios",
    "Naive Bayes": "Medium - per-feature likelihoods, strong independence assumption",
    "Random Forest": "Low - ensemble of trees (feature importances only)",
}


# ----------------------------------------------------------------------------
# 5. FULL STUDY (identical held-out test set for every algorithm)
# ----------------------------------------------------------------------------
def make_splits(df, seed=42, test_size=0.2, val_size=0.2, target=TARGET):
    df = df.drop_duplicates().reset_index(drop=True)        # identical modules would leak across splits
    X, y = df.drop(columns=[target]), df[target].values
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=val_size / (1 - test_size),
                                          stratify=ytv, random_state=seed)
    return dict(Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=ytr, yva=yva, yte=yte, n_dedup=len(df))


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def run_study(df, seed=42, test_size=0.2, val_size=0.2, temperature=1.0, max_features=10,
              n_draws=2000, fast=False, target=TARGET) -> dict:
    sp = make_splits(df, seed, test_size, val_size, target)
    pre = Preprocessor().fit(sp["Xtr"])
    Ztr, Zva, Zte = (pre.transform(sp[k]) for k in ("Xtr", "Xva", "Xte"))
    ytr, yva, yte = sp["ytr"], sp["yva"], sp["yte"]
    names = pre.columns_
    probs, fit_t, inf_t, extra = {}, {}, {}, {}

    # --- Bayesian Model Averaging (no validation data needed to fit) ---
    bma, t = _timed(lambda: BMAClassifier(max_features=max_features, temperature=temperature,
                                          n_draws=n_draws, random_state=seed).fit(Ztr, ytr, names))
    fit_t["Bayesian Model Averaging"] = t
    pv = bma.predict_proba(Zva)
    pt, ti = _timed(lambda: bma.predict_proba(Zte))
    probs["Bayesian Model Averaging"] = (pv, pt)
    inf_t["Bayesian Model Averaging"] = ti
    probs["Best single BIC model"] = (bma.predict_proba_best(Zva), bma.predict_proba_best(Zte))
    fit_t["Best single BIC model"] = t
    inf_t["Best single BIC model"] = 0.0

    # --- Logistic Regression (C tuned on validation log-loss) ---
    def fit_lr():
        best = None
        for C in (0.01, 0.1, 1, 10, 100):
            m = LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr)
            s = log_loss(yva, m.predict_proba(Zva)[:, 1])
            if best is None or s < best[0]:
                best = (s, m, C)
        return best
    (_, lr, lrC), t = _timed(fit_lr)
    fit_t["Logistic Regression"] = t
    pt, ti = _timed(lambda: lr.predict_proba(Zte)[:, 1])
    probs["Logistic Regression"] = (lr.predict_proba(Zva)[:, 1], pt)
    inf_t["Logistic Regression"] = ti
    extra["lr"], extra["lr_C"] = lr, lrC

    # --- Naive Bayes (var_smoothing tuned on validation log-loss) ---
    def fit_nb():
        best = None
        for vs in (1e-9, 1e-6, 1e-3, 1e-1):
            m = GaussianNB(var_smoothing=vs).fit(Ztr, ytr)
            s = log_loss(yva, np.clip(m.predict_proba(Zva)[:, 1], 1e-6, 1 - 1e-6))
            if best is None or s < best[0]:
                best = (s, m)
        return best
    (_, nb), t = _timed(fit_nb)
    fit_t["Naive Bayes"] = t
    pt, ti = _timed(lambda: nb.predict_proba(Zte)[:, 1])
    probs["Naive Bayes"] = (nb.predict_proba(Zva)[:, 1], pt)
    inf_t["Naive Bayes"] = ti

    # --- Random Forest (small grid tuned on validation log-loss) ---
    def fit_rf():
        grid = [(6, 5)] if fast else [(4, 5), (8, 3), (None, 2), (None, 5)]
        best = None
        for depth, leaf in grid:
            m = RandomForestClassifier(n_estimators=150 if fast else 250, max_depth=depth,
                                       min_samples_leaf=leaf, n_jobs=-1, random_state=seed).fit(Ztr, ytr)
            s = log_loss(yva, np.clip(m.predict_proba(Zva)[:, 1], 1e-3, 1 - 1e-3))
            if best is None or s < best[0]:
                best = (s, m)
        return best
    (_, rf), t = _timed(fit_rf)
    fit_t["Random Forest"] = t
    pt, ti = _timed(lambda: rf.predict_proba(Zte)[:, 1])
    probs["Random Forest"] = (rf.predict_proba(Zva)[:, 1], pt)
    inf_t["Random Forest"] = ti
    extra["rf"] = rf

    # --- comparison table ---
    rows, thr = [], {}
    for name, (pv, pt) in probs.items():
        thr[name] = best_f1_threshold(yva, pv)
        mv, mt = metrics_at(yva, pv, thr[name]), metrics_at(yte, pt, thr[name])
        rows.append({"Algorithm": name, "Validation ROC-AUC": mv["ROC_AUC"], "Test ROC-AUC": mt["ROC_AUC"],
                     "Test PR-AUC": mt["PR_AUC"], "Test F1": mt["F1"], "Test Precision": mt["Precision"],
                     "Test Recall": mt["Recall"], "Test Brier": mt["Brier"], "Test Log-loss": mt["LogLoss"],
                     "Test ECE": mt["ECE"], "Training time (s)": fit_t[name],
                     "Inference time (ms/1000)": 1000 * inf_t[name] / len(yte) * 1000,
                     "Interpretability": INTERPRETABILITY[name]})
    table = pd.DataFrame(rows)
    return dict(table=table, probs=probs, thr=thr, bma=bma, pre=pre, splits=sp, extra=extra,
                Ztr=Ztr, Zva=Zva, Zte=Zte, ytr=ytr, yva=yva, yte=yte, seed=seed)


def repeated_study(df, seeds=range(5), **kw) -> pd.DataFrame:
    """Re-run the whole study on several random splits (robustness)."""
    out = []
    for s in seeds:
        r = run_study(df, seed=int(s), fast=True, **kw)
        t = r["table"].copy()
        t["seed"] = s
        out.append(t)
    return pd.concat(out, ignore_index=True)


def summarise_repeated(rep: pd.DataFrame, cols=("Test ROC-AUC", "Test F1", "Test Brier", "Test Log-loss", "Test ECE")):
    g = rep.groupby("Algorithm")[list(cols)]
    m, s = g.mean(), g.std()
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(m[c], s[c])]
    return out.reset_index()


def score_modules(result: dict, raw_df: pd.DataFrame, level=0.90, n_draws=1000) -> pd.DataFrame:
    """Score arbitrary raw modules with BMA -> probability, credible interval, uncertainty, tier."""
    Z = result["pre"].transform(raw_df)
    d = result["bma"].predict_dist(Z, level=level, n_draws=n_draws)
    thr = result["thr"]["Bayesian Model Averaging"]
    out = pd.DataFrame({"p_defect": d["mean"], "ci_low": d["lo"], "ci_high": d["hi"],
                        "ci_width": d["width"], "epistemic_uncertainty": d["epistemic"],
                        "aleatoric_uncertainty": d["aleatoric"]}, index=raw_df.index)
    out["risk_tier"] = np.where(out.ci_low >= thr, "High", np.where(out.ci_high < thr, "Low", "Review"))
    return out


if __name__ == "__main__":
    d = load_dataset("KC1")
    r = run_study(d)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(r["table"].drop(columns="Interpretability").round(4))
    print(r["bma"].weights_table().head(8))
