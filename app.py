"""
DefectLens - Bayesian Model Averaging for software defect prediction
Glassmorphism Streamlit demo.      Run:  streamlit run app.py
"""
import hashlib
import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bma as B  # noqa: E402

st.set_page_config(page_title="DefectLens | Bayesian defect prediction", page_icon="🔮",
                   layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# THEME  (dark violet glassmorphism, inspired by the Reflect landing page)
# =============================================================================
VIOLET, LILAC, CYAN, PINK, AMBER, GREEN = "#8b5cf6", "#c4b5fd", "#22d3ee", "#f472b6", "#fbbf24", "#34d399"
TIER_COLOR = {"High": PINK, "Review": AMBER, "Low": CYAN}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, .stApp, button, input, textarea, .stMarkdown, .stMarkdown *, [data-testid="stTab"] *, .stSlider *, label, .stSelectbox * { font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important; }
.stApp {
  background:
    radial-gradient(1100px 520px at 50% -8%, rgba(99,52,214,.30), transparent 60%),
    radial-gradient(700px 420px at 90% 38%, rgba(34,211,238,.07), transparent 65%),
    radial-gradient(800px 500px at 8% 62%, rgba(168,85,247,.12), transparent 65%),
    linear-gradient(180deg, #05030f 0%, #070416 45%, #04020c 100%);
  color: #e9e5ff;
}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { max-width: 1260px; padding-top: 1.1rem; padding-bottom: 4rem; }
h1,h2,h3,h4 { color:#f5f3ff; letter-spacing:-.02em; }
p, li, label, span, div { color: inherit; }
hr { border-color: rgba(255,255,255,.08) !important; }

/* ---------- glass primitives ---------- */
.glass, div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-title) {
  background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.022)) !important;
  border: 1px solid rgba(255,255,255,.11) !important; border-radius: 22px !important;
  backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
  box-shadow: 0 10px 50px rgba(76,29,200,.16), inset 0 1px 0 rgba(255,255,255,.10);
  padding: 14px 16px 10px !important;
}
.card-title { font-weight:600; font-size:1.02rem; color:#f5f3ff; margin:.2rem 0 .1rem; letter-spacing:-.01em; }
.card-sub { font-size:.82rem; color:#a79fd0; margin-bottom:.4rem; line-height:1.45; }

/* ---------- top nav ---------- */
.nav { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:6px; }
.brand { display:flex; align-items:center; gap:11px; font-weight:700; font-size:1.08rem; color:#fff; letter-spacing:-.02em; }
.logo { width:38px; height:38px; border-radius:11px; background: radial-gradient(circle at 30% 25%, #a78bfa, #4c1d95 70%);
  box-shadow: 0 0 22px rgba(139,92,246,.65), inset 0 0 0 1px rgba(255,255,255,.25); display:grid; place-items:center; }
.logo i { width:18px; height:18px; border-radius:50%; border:2px solid rgba(255,255,255,.9); box-shadow:0 0 10px rgba(255,255,255,.7) inset; }
.navpill { padding:9px 22px; border-radius:999px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.10);
  backdrop-filter: blur(18px); font-size:.86rem; color:#cfc8ee; display:flex; gap:18px; align-items:center; }
.navpill b { color:#fff; font-weight:600; }
.dot { width:8px; height:8px; border-radius:50%; background:#34d399; box-shadow:0 0 10px #34d399; display:inline-block; margin-right:7px; }
.cta { padding:10px 20px; border-radius:12px; background:linear-gradient(180deg, rgba(139,92,246,.35), rgba(91,33,182,.35));
  border:1px solid rgba(167,139,250,.45); color:#fff; font-weight:600; font-size:.87rem; box-shadow:0 0 24px rgba(139,92,246,.35); }

/* ---------- hero ---------- */
.hero { text-align:center; padding: 40px 0 0; position:relative; }
.badge { display:inline-flex; gap:9px; align-items:center; padding:8px 16px; border-radius:999px; font-size:.82rem; color:#d9d3ff;
  background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.11); backdrop-filter: blur(14px); }
.hero h1 { font-size: clamp(2.4rem, 5.4vw, 4.4rem); font-weight:800; line-height:1.04; margin:22px 0 14px;
  background: linear-gradient(180deg, #ffffff 25%, #b9acf5 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p.sub { color:#b7aedf; font-size:1.08rem; margin:0 auto; max-width:640px; }
.halo { position:relative; height:250px; margin-top:6px; overflow:hidden; }
.halo .glow { position:absolute; left:50%; bottom:-170px; width:980px; height:420px; transform:translateX(-50%);
  background: radial-gradient(ellipse at 50% 100%, rgba(170,140,255,.85) 0%, rgba(124,58,237,.45) 28%, rgba(76,29,149,.18) 52%, transparent 72%);
  filter: blur(26px); animation: breathe 7s ease-in-out infinite; }
.halo .arc { position:absolute; left:50%; bottom:-2px; transform:translateX(-50%); border-radius: 50% 50% 0 0 / 100% 100% 0 0; border-bottom:none; }
.halo .a3 { width:640px; height:215px; border:2px solid rgba(167,139,250,.28); box-shadow:0 0 50px rgba(139,92,246,.4); }
.halo .a2 { width:470px; height:160px; border:4px solid rgba(221,214,254,.75); box-shadow:0 0 36px 4px rgba(167,139,250,.85), inset 0 0 30px rgba(167,139,250,.55); }
.halo .a1 { width:320px; height:108px; border:5px solid #fff; box-shadow:0 0 30px 6px rgba(255,255,255,.55), 0 0 70px 14px rgba(167,139,250,.9), inset 0 0 30px rgba(196,181,253,.9);
  background: radial-gradient(ellipse at 50% 100%, rgba(124,58,237,.9), rgba(30,10,80,.9) 70%); }
.halo .line { position:absolute; left:6%; right:6%; bottom:0; height:2px; background:linear-gradient(90deg, transparent, rgba(196,181,253,.9), transparent); box-shadow:0 0 22px 4px rgba(167,139,250,.7); }
@keyframes breathe { 0%,100% { opacity:.85; transform:translateX(-50%) scale(1);} 50% { opacity:1; transform:translateX(-50%) scale(1.06);} }

/* ---------- KPI bar ---------- */
.kpibar { display:grid; grid-template-columns: repeat(5, 1fr); gap:0; border-radius:20px; padding:16px 8px; margin: -34px 0 26px;
  background: linear-gradient(180deg, rgba(255,255,255,.085), rgba(255,255,255,.03)); border:1px solid rgba(255,255,255,.12);
  backdrop-filter: blur(24px); box-shadow: 0 18px 60px rgba(76,29,200,.30), inset 0 1px 0 rgba(255,255,255,.12); position:relative; z-index:2; }
.kpi { text-align:center; padding: 4px 8px; border-right:1px solid rgba(255,255,255,.07); }
.kpi:last-child { border-right:none; }
.kpi .v { font-size:1.65rem; font-weight:700; color:#fff; letter-spacing:-.02em; }
.kpi .l { font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:#9f96cc; margin-top:2px; }
.kpi .d { font-size:.74rem; color:#34d399; }

/* ---------- stat cards ---------- */
.stat { padding:18px 20px; border-radius:20px; background:linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
  border:1px solid rgba(255,255,255,.10); backdrop-filter: blur(20px); box-shadow:0 8px 40px rgba(76,29,200,.12), inset 0 1px 0 rgba(255,255,255,.09); }
.stat .l { font-size:.76rem; color:#a79fd0; text-transform:uppercase; letter-spacing:.08em; }
.stat .v { font-size:2.0rem; font-weight:700; color:#fff; letter-spacing:-.02em; margin-top:4px; }
.stat .s { font-size:.8rem; color:#8f86bd; }
.pill { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.74rem; font-weight:600; }
.p-High { background:rgba(244,114,182,.16); color:#f9a8d4; border:1px solid rgba(244,114,182,.35); }
.p-Review { background:rgba(251,191,36,.14); color:#fcd34d; border:1px solid rgba(251,191,36,.35); }
.p-Low { background:rgba(34,211,238,.13); color:#67e8f9; border:1px solid rgba(34,211,238,.32); }

/* ---------- tables ---------- */
.gt-wrap { overflow-x:auto; border-radius:16px; border:1px solid rgba(255,255,255,.08); background:rgba(8,5,24,.35); }
table.gt { width:100%; border-collapse:collapse; font-size:.84rem; }
table.gt th { text-align:left; padding:11px 14px; color:#b4abe0; font-weight:600; font-size:.74rem; text-transform:uppercase; letter-spacing:.07em;
  background:rgba(255,255,255,.04); border-bottom:1px solid rgba(255,255,255,.08); white-space:nowrap; }
table.gt td { padding:10px 14px; border-bottom:1px solid rgba(255,255,255,.05); color:#e7e3ff; white-space:nowrap; }
table.gt tr:hover td { background:rgba(139,92,246,.09); }
table.gt td.best { color:#fff; font-weight:700; background:rgba(139,92,246,.17); }
table.gt td.wrap { white-space:normal; min-width:240px; color:#b7aedf; }

/* ---------- streamlit widgets ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(14,8,40,.92), rgba(6,3,18,.95)); border-right:1px solid rgba(255,255,255,.08);
  backdrop-filter: blur(24px); }
section[data-testid="stSidebar"] * { color:#ddd8f7; }
[data-testid="stTabs"] [role="tablist"] { gap:2px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.11); border-radius:999px;
  padding:6px; width:fit-content; margin:0 auto 24px; backdrop-filter: blur(20px); flex-wrap:wrap; justify-content:center; box-shadow:0 8px 40px rgba(76,29,200,.2); }
[data-testid="stTabs"] [role="tablist"] ~ * , [data-testid="stTabs"] > div:first-child { border-bottom:none !important; }
[data-testid="stTab"] { border-radius:999px; padding:9px 15px; height:auto; color:#b4abe0; font-weight:500; background:transparent; border:none; }
[data-testid="stTab"] p { color:inherit; font-size:.92rem; margin:0; }
[data-testid="stTab"][aria-selected="true"] { background:linear-gradient(180deg, rgba(139,92,246,.6), rgba(91,33,182,.55)) !important; color:#fff !important;
  box-shadow:0 0 22px rgba(139,92,246,.55), inset 0 1px 0 rgba(255,255,255,.28); }
.react-aria-SelectionIndicator { display:none !important; }
[data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] [role="tablist"]::before { display:none !important; }
.stButton>button, .stDownloadButton>button { border-radius:12px; border:1px solid rgba(167,139,250,.45); color:#fff; font-weight:600;
  background:linear-gradient(180deg, rgba(139,92,246,.40), rgba(91,33,182,.38)); box-shadow:0 0 22px rgba(139,92,246,.3); }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#c4b5fd; box-shadow:0 0 30px rgba(167,139,250,.6); color:#fff; }
div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input { background:rgba(255,255,255,.05) !important; border:1px solid rgba(255,255,255,.12) !important; border-radius:12px !important; color:#fff !important; }
[data-testid="stSlider"] [role="slider"] { background:#a78bfa !important; box-shadow:0 0 14px #8b5cf6; }
[data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.04); border:1px dashed rgba(167,139,250,.4); border-radius:14px; }
[data-testid="stMetric"] { background:rgba(255,255,255,.04); border-radius:14px; padding:10px 14px; border:1px solid rgba(255,255,255,.08); }
.stRadio [role="radiogroup"] { gap:8px; }
.footer { text-align:center; color:#6f67a0; font-size:.8rem; margin-top:34px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================
def H(html: str):
    st.markdown("\n".join(l.strip() for l in html.strip().splitlines()), unsafe_allow_html=True)


def card_head(title, sub=""):
    H(f'<div class="card-title">{title}</div><div class="card-sub">{sub}</div>')


def style_fig(fig, h=360, legend=True, **kw):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cfc8ee", size=12), margin=dict(l=10, r=10, t=30, b=10),
        colorway=[VIOLET, CYAN, PINK, AMBER, GREEN, LILAC],
        legend=dict(bgcolor="rgba(255,255,255,.04)", bordercolor="rgba(255,255,255,.08)", borderwidth=1,
                    font=dict(size=11), orientation="h", y=1.14, x=0) if legend else dict(visible=False),
        hoverlabel=dict(bgcolor="#1b1140", bordercolor="#8b5cf6", font=dict(color="#fff")), **kw)
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.10)", linecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.10)", linecolor="rgba(255,255,255,.12)")
    return fig


def plot(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def gtable(df, fmt=None, best=None, wrap=(), raw=()):
    """Glass HTML table. best = {col: 'max'|'min'} highlights the winner."""
    fmt = fmt or {}
    best = best or {}
    win = {c: (df[c].idxmax() if m == "max" else df[c].idxmin()) for c, m in best.items() if c in df}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for i, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if c in raw:
                s = v
            elif c in fmt and isinstance(v, (int, float, np.floating, np.integer)):
                s = fmt[c].format(v)
            else:
                s = v
            cls = "best" if win.get(c) == i else ("wrap" if c in wrap else "")
            tds.append(f'<td class="{cls}">{s}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    H(f'<div class="gt-wrap"><table class="gt"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def pill(t):
    return f'<span class="pill p-{t}">{t}</span>'


# =============================================================================
# DATA + MODEL (cached)
# =============================================================================
def to_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin(["y", "yes", "true", "1", "defective", "buggy"]).astype(int)


@st.cache_data(show_spinner=False)
def get_dataset(name):
    return B.load_dataset(name)


@st.cache_resource(show_spinner=False)
def get_study(key, df_bytes, seed, test_size, tau, max_features):
    df = pd.read_pickle(io.BytesIO(df_bytes))
    return B.run_study(df, seed=seed, test_size=test_size, val_size=0.2, temperature=tau,
                       max_features=max_features, n_draws=2000)


@st.cache_data(show_spinner=False)
def get_repeated(key, df_bytes, n, tau):
    df = pd.read_pickle(io.BytesIO(df_bytes))
    rep = B.repeated_study(df, seeds=range(n), temperature=tau)
    return rep


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    H('<div class="brand" style="margin-bottom:14px"><div class="logo"><i></i></div>DefectLens</div>')
    st.caption("Bayesian Model Averaging · software defect prediction")
    src = st.radio("Data source", ["NASA / PROMISE dataset", "Upload my CSV"], horizontal=False)
    if src.startswith("NASA"):
        ds = st.selectbox("Dataset", list(B.DATASETS), index=3, format_func=lambda k: f"{k} · {B.DATASETS[k].split(' - ')[1]}")
        try:
            raw = get_dataset(ds)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load {ds}: {e}")
            st.stop()
        data_key = ds
    else:
        up = st.file_uploader("CSV with code metrics + a binary defect column", type="csv")
        if up is None:
            st.info("Upload a CSV, or switch back to a NASA dataset.")
            st.stop()
        udf = pd.read_csv(up)
        tgt = st.selectbox("Defect column", [c for c in udf.columns[::-1]])
        feats = udf.drop(columns=[tgt]).select_dtypes("number")
        raw = feats.copy()
        raw[B.TARGET] = to_binary(udf[tgt]).values
        data_key = "upload-" + hashlib.md5(pd.util.hash_pandas_object(raw).values.tobytes()).hexdigest()[:10]
    st.markdown("---")
    seed = st.number_input("Random seed (split)", 0, 9999, 42, 1)
    test_size = st.slider("Held-out test share", 0.15, 0.40, 0.20, 0.05)
    tau = st.slider("Posterior temperature τ", 0.10, 1.00, 1.00, 0.05,
                    help="τ=1 is the textbook BIC posterior. Lower τ flattens the model weights -> more model uncertainty.")
    max_f = st.slider("Max metrics per model", 3, 14, 10)
    level = st.select_slider("Credible interval", [0.80, 0.90, 0.95], value=0.90, format_func=lambda v: f"{int(v*100)}%")
    st.markdown("---")
    st.caption("Same held-out test set for every algorithm · preprocessing fitted on training data only · "
               "thresholds & hyper-parameters tuned on validation data only.")

buf = io.BytesIO()
raw.to_pickle(buf)
with st.spinner("Fitting Bayesian candidate models and baselines …"):
    R = get_study(data_key, buf.getvalue(), int(seed), float(test_size), float(tau), int(max_f))

bma, pre, sp, tab = R["bma"], R["pre"], R["splits"], R["table"]
thr = R["thr"]["Bayesian Model Averaging"]
row = tab.set_index("Algorithm")
b = row.loc["Bayesian Model Averaging"]
w = bma.weights_
n_eff = 1.0 / np.sum(w ** 2)
n_all = sp["n_dedup"]
rate = (np.concatenate([sp["ytr"], sp["yva"], sp["yte"]])).mean()

# =============================================================================
# NAV + HERO
# =============================================================================
H(f"""
<div class="nav">
  <div class="brand"><div class="logo"><i></i></div>DefectLens</div>
  <div class="navpill"><span><span class="dot"></span><b>{data_key if not data_key.startswith('upload') else 'Your data'}</b></span>
    <span>{n_all:,} unique modules</span><span>{rate*100:.1f}% defective</span><span>{len(pre.columns_)} metrics kept</span></div>
  <div class="cta">Bayesian Model Averaging</div>
</div>
<div class="hero">
  <div class="badge"><span style="color:#c4b5fd">✦</span> New: posterior model weights &amp; credible intervals, live</div>
  <h1>Predict defects.<br>Quantify the doubt.</h1>
  <p class="sub">Never ship a risky module blind. DefectLens averages hundreds of Bayesian models and tells you
  not just <i>what</i> is likely to fail — but how sure it is.</p>
  <div class="halo"><div class="glow"></div><div class="arc a3"></div><div class="arc a2"></div><div class="arc a1"></div><div class="line"></div></div>
</div>
<div class="kpibar">
  <div class="kpi"><div class="v">{b['Test ROC-AUC']:.3f}</div><div class="l">Test ROC-AUC</div></div>
  <div class="kpi"><div class="v">{b['Test F1']:.3f}</div><div class="l">Test F1</div></div>
  <div class="kpi"><div class="v">{b['Test Brier']:.3f}</div><div class="l">Brier score</div></div>
  <div class="kpi"><div class="v">{b['Test ECE']:.3f}</div><div class="l">Calibration error</div></div>
  <div class="kpi"><div class="v">{n_eff:.1f}</div><div class="l">Effective models</div></div>
</div>
""")

# scored modules (held-out test set by default)
scope = "Held-out test set"
Xte_raw = sp["Xte"]
S_test = B.score_modules(R, Xte_raw, level=level, n_draws=1000)
dist_test = bma.predict_dist(R["Zte"], level=level, n_draws=1000, seed=7)

tabs = st.tabs(["◉ Risk dashboard", "⚖ Model weights", "≈ Uncertainty", "◐ Calibration", "▤ Comparison", "⚡ Playground", "ⓘ About"])

# -----------------------------------------------------------------------------
# TAB 1 - RISK DASHBOARD
# -----------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    cnt = S_test.risk_tier.value_counts()
    for col, label, sub, val in [
        (c1, "High risk", "credible interval fully above threshold", cnt.get("High", 0)),
        (c2, "Needs review", "interval straddles the threshold", cnt.get("Review", 0)),
        (c3, "Low risk", "interval fully below threshold", cnt.get("Low", 0)),
        (c4, f"Avg {int(level*100)}% interval width", "mean predictive uncertainty", f"{S_test.ci_width.mean():.2f}")]:
        col.markdown(f'<div class="stat"><div class="l">{label}</div><div class="v">{val}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        with st.container(border=True):
            card_head("Top-20 riskiest modules", f"Posterior mean P(defect) with {int(level*100)}% credible interval · ◯ = actually defective")
            top = S_test.sort_values("p_defect", ascending=False).head(20).iloc[::-1]
            lab = [f"module #{i}" for i in top.index]
            yv = pd.Series(sp["yte"], index=Xte_raw.index).loc[top.index].values
            fig = go.Figure()
            for t in ["High", "Review", "Low"]:
                m = (top.risk_tier == t).values
                if m.any():
                    fig.add_trace(go.Scatter(x=top.p_defect[m], y=np.array(lab)[m], mode="markers", name=f"{t}",
                        marker=dict(color=TIER_COLOR[t], size=10, line=dict(width=1, color="#fff")),
                        error_x=dict(type="data", symmetric=False, array=(top.ci_high - top.p_defect)[m],
                                     arrayminus=(top.p_defect - top.ci_low)[m], color=TIER_COLOR[t], thickness=2.2, width=0),
                        hovertemplate="%{y}<br>P(defect)=%{x:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=top.p_defect[yv == 1], y=np.array(lab)[yv == 1], mode="markers", name="actual defect",
                                     marker=dict(symbol="circle-open", size=19, color="#fff", line=dict(width=1.6)), hoverinfo="skip"))
            fig.add_vline(x=thr, line_dash="dot", line_color=LILAC, annotation_text=f"threshold {thr:.2f}", annotation_font_color=LILAC)
            style_fig(fig, 520)
            fig.update_xaxes(range=[0, 1.02], title="P(defect)")
            plot(fig, "top20")
    with right:
        with st.container(border=True):
            card_head("Risk vs. uncertainty map", "Each dot is a held-out module — high & wide = risky and unsure")
            fig = go.Figure()
            for t in ["High", "Review", "Low"]:
                m = S_test.risk_tier == t
                fig.add_trace(go.Scatter(x=S_test.p_defect[m], y=S_test.ci_width[m], mode="markers", name=t,
                    marker=dict(color=TIER_COLOR[t], size=7, opacity=.75, line=dict(width=0)),
                    hovertemplate="P=%{x:.2f}<br>width=%{y:.2f}<extra></extra>"))
            fig.add_vline(x=thr, line_dash="dot", line_color=LILAC)
            style_fig(fig, 250)
            fig.update_xaxes(title="P(defect)", range=[0, 1])
            fig.update_yaxes(title="interval width")
            plot(fig, "riskmap")
            card_head("Risk-tier mix", "")
            tiers = S_test.risk_tier.value_counts().reindex(["High", "Review", "Low"]).fillna(0)
            fig = go.Figure(go.Bar(x=tiers.values, y=tiers.index, orientation="h", marker_color=[TIER_COLOR[t] for t in tiers.index],
                                   text=tiers.values.astype(int), textposition="outside", textfont=dict(color="#fff")))
            style_fig(fig, 150, legend=False)
            fig.update_xaxes(range=[0, tiers.max() * 1.2])
            plot(fig, "tiers")
    with st.container(border=True):
        card_head("Module risk table", "Sorted by posterior defect probability — download the full scored list below")
        show = S_test.copy()
        show["actual"] = np.where(pd.Series(sp["yte"], index=Xte_raw.index).loc[show.index] == 1, "defective", "clean")
        show = show.sort_values("p_defect", ascending=False)
        keymetrics = [c for c in ["LOC_TOTAL", "CYCLOMATIC_COMPLEXITY"] if c in Xte_raw.columns] or list(pre.columns_[:2])
        t = pd.DataFrame({"Module": [f"#{i}" for i in show.index[:15]],
                          "P(defect)": [f"{v*100:.1f}%" for v in show.p_defect[:15]],
                          f"{int(level*100)}% interval": [f"{a*100:.0f}% – {c*100:.0f}%" for a, c in zip(show.ci_low[:15], show.ci_high[:15])],
                          "Tier": [pill(x) for x in show.risk_tier[:15]],
                          "Epistemic": [f"{v:.3f}" for v in show.epistemic_uncertainty[:15]],
                          "Truth": show.actual[:15].values})
        for k in keymetrics:
            t[k] = Xte_raw.loc[show.index[:15], k].round(1).values
        gtable(t, raw=("Tier",))
        out = show.join(Xte_raw)
        st.download_button("⬇ Download scored modules (CSV)", out.to_csv().encode(), "defectlens_scored_modules.csv", "text/csv")

# -----------------------------------------------------------------------------
# TAB 2 - MODEL WEIGHTS
# -----------------------------------------------------------------------------
with tabs[1]:
    wt = bma.weights_table()
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [(c1, "Candidate models", len(bma.models_), f"{bma.n_visited_} subsets visited by MC3"),
                                 (c2, "Effective models", f"{n_eff:.1f}", "1 / Σ weight²"),
                                 (c3, "Top model weight", f"{w.max()*100:.1f}%", bma.best_model_["name"]),
                                 (c4, "Occam's window", f"{(w>0.01).sum()}", "models with weight > 1%")]:
        col.markdown(f'<div class="stat"><div class="l">{label}</div><div class="v">{val}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)
    st.write("")
    l, r = st.columns(2)
    with l:
        with st.container(border=True):
            card_head("Posterior model weights  P(Mₖ | data)", "∝ exp(−τ·BIC/2) — Schwarz approximation to the marginal likelihood")
            top = wt.head(14).iloc[::-1]
            colr = {"logreg": VIOLET, "nb": CYAN, "tree": PINK}
            fig = go.Figure(go.Bar(x=top.posterior_weight, y=top.model, orientation="h", marker_color=[colr[k] for k in top.kind],
                                   customdata=np.c_[top.params, top.BIC.round(1)],
                                   hovertemplate="%{y}<br>weight %{x:.3f}<br>params %{customdata[0]} · BIC %{customdata[1]}<extra></extra>"))
            style_fig(fig, 430, legend=False)
            fig.update_xaxes(title="posterior weight")
            plot(fig, "weights")
    with r:
        with st.container(border=True):
            card_head("Posterior inclusion probability of each metric", "Share of posterior mass on models that include the metric")
            pip = bma.inclusion_probabilities().head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=pip.values, y=pip.index, orientation="h",
                                   marker=dict(color=pip.values, colorscale=[[0, "#3b1d8f"], [1, "#e9d5ff"]], line=dict(width=0))))
            style_fig(fig, 430, legend=False)
            fig.update_xaxes(range=[0, 1.02], title="inclusion probability")
            plot(fig, "pip")
    with st.container(border=True):
        card_head("Model leaderboard", "Complexity penalty comes from BIC = −2·logL + k·ln n")
        lt = wt.head(10).copy()
        lt["posterior_weight"] = (lt.posterior_weight * 100).map("{:.2f}%".format)
        lt["log_likelihood"] = lt.log_likelihood.map("{:.1f}".format)
        lt["BIC"] = lt.BIC.map("{:.1f}".format)
        lt["metrics"] = lt.metrics.map(lambda s: (s[:70] + "…") if len(s) > 70 else s)
        lt = lt.rename(columns={"model": "Model", "kind": "Family", "params": "k", "log_likelihood": "log-lik", "posterior_weight": "Weight", "metrics": "Metrics in model"})
        gtable(lt, wrap=("Metrics in model",))

# -----------------------------------------------------------------------------
# TAB 3 - UNCERTAINTY
# -----------------------------------------------------------------------------
with tabs[2]:
    yte = R["yte"]
    l, r = st.columns(2)
    with l:
        with st.container(border=True):
            card_head("Aleatoric vs. epistemic uncertainty", "Epistemic = what more data / better models could reduce (mutual information)")
            fig = go.Figure(go.Scatter(x=dist_test["aleatoric"], y=dist_test["epistemic"], mode="markers",
                                       marker=dict(size=7, color=dist_test["mean"], colorscale=[[0, CYAN], [.5, VIOLET], [1, PINK]],
                                                   showscale=True, colorbar=dict(title="P(defect)", thickness=10), opacity=.8, line=dict(width=0)),
                                       hovertemplate="aleatoric %{x:.3f}<br>epistemic %{y:.3f}<extra></extra>"))
            style_fig(fig, 360, legend=False)
            fig.update_xaxes(title="aleatoric (expected entropy, nats)")
            fig.update_yaxes(title="epistemic (mutual information)")
            plot(fig, "unc1")
    with r:
        with st.container(border=True):
            card_head("Where does the variance come from?", "Between-model (which model?) vs. within-model (which coefficients?)")
            bv, wv = dist_test["between_var"].mean(), dist_test["within_var"].mean()
            fig = go.Figure(go.Bar(x=["between-model", "within-model (parameters)"], y=[bv, wv], marker_color=[VIOLET, CYAN],
                                   text=[f"{bv:.4f}", f"{wv:.4f}"], textposition="outside", textfont=dict(color="#fff")))
            style_fig(fig, 360, legend=False)
            fig.update_yaxes(title="mean predictive variance")
            plot(fig, "unc2")
    with st.container(border=True):
        card_head("Uncertainty-aware triage (selective prediction)",
                  "Rank held-out modules by epistemic uncertainty, hand the most uncertain ones to a human reviewer, and score the rest")
        order = np.argsort(dist_test["epistemic"])
        fr = np.arange(1.0, 0.45, -0.05)
        auc_c, br_c = [], []
        for f in fr:
            keep = order[: max(int(round(f * len(order))), 10)]
            yy, pp = yte[keep], dist_test["mean"][keep]
            auc_c.append(roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan)
            br_c.append(brier_score_loss(yy, pp))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fr * 100, y=auc_c, mode="lines+markers", name="ROC-AUC (higher is better)", line=dict(color=VIOLET, width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=fr * 100, y=br_c, mode="lines+markers", name="Brier (lower is better)", line=dict(color=PINK, width=3), marker=dict(size=8), yaxis="y2"))
        style_fig(fig, 340)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)", title="Brier"))
        fig.update_xaxes(title="% of modules auto-scored (most certain first)", autorange="reversed")
        fig.update_yaxes(title="ROC-AUC")
        plot(fig, "triage")
    with st.container(border=True):
        card_head("Posterior predictive of a single module", "The full distribution of P(defect) across models and parameter draws")
        order_p = S_test.sort_values("p_defect", ascending=False).index.tolist()
        pick = st.selectbox("Module", order_p[:60], format_func=lambda i: f"#{i} · P={S_test.p_defect[i]*100:.1f}% · {S_test.risk_tier[i]}")
        j = Xte_raw.index.get_loc(pick)
        dr = dist_test["draws"][j]
        fig = go.Figure(go.Histogram(x=dr, nbinsx=40, marker=dict(color=VIOLET, line=dict(width=0)), opacity=.85))
        for xv, nm, cl in [(dist_test["mean"][j], "mean", "#fff"), (dist_test["lo"][j], "lower", LILAC), (dist_test["hi"][j], "upper", LILAC), (thr, "decision threshold", AMBER)]:
            fig.add_vline(x=xv, line_dash="dash" if nm != "mean" else "solid", line_color=cl, annotation_text=nm, annotation_font_color=cl)
        style_fig(fig, 300, legend=False)
        fig.update_xaxes(title="P(defect)", range=[0, 1])
        plot(fig, "single")

# -----------------------------------------------------------------------------
# TAB 4 - CALIBRATION
# -----------------------------------------------------------------------------
with tabs[3]:
    l, r = st.columns([1.2, 1])
    with l:
        with st.container(border=True):
            card_head("Reliability diagram (held-out test set)", "A perfectly calibrated model sits on the diagonal")
            fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="rgba(255,255,255,.35)", dash="dash"), name="perfect", hoverinfo="skip"))
            cmap = {"Bayesian Model Averaging": VIOLET, "Logistic Regression": CYAN, "Random Forest": AMBER, "Naive Bayes": PINK}
            for nm, cl in cmap.items():
                pt = R["probs"][nm][1]
                px, py, _ = B.reliability_curve(R["yte"], pt, 6)
                fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", name=nm, line=dict(color=cl, width=3 if nm.startswith("Bayes") else 2), marker=dict(size=8)))
            style_fig(fig, 430)
            fig.update_xaxes(title="mean predicted probability", range=[0, 1])
            fig.update_yaxes(title="observed defect frequency", range=[0, 1])
            plot(fig, "rel")
    with r:
        with st.container(border=True):
            card_head("Proper scoring rules", "Lower is better")
            ct = tab[["Algorithm", "Test Brier", "Test Log-loss", "Test ECE"]].copy()
            gtable(ct, fmt={c: "{:.3f}" for c in ct.columns[1:]}, best={c: "min" for c in ct.columns[1:]})
            st.write("")
            card_head("Distribution of BMA probabilities", "")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=R["probs"]["Bayesian Model Averaging"][1][R["yte"] == 0], name="clean", marker_color=CYAN, opacity=.7, nbinsx=25))
            fig.add_trace(go.Histogram(x=R["probs"]["Bayesian Model Averaging"][1][R["yte"] == 1], name="defective", marker_color=PINK, opacity=.7, nbinsx=25))
            fig.update_layout(barmode="overlay")
            style_fig(fig, 190)
            plot(fig, "hist")

# -----------------------------------------------------------------------------
# TAB 5 - COMPARISON
# -----------------------------------------------------------------------------
with tabs[4]:
    with st.container(border=True):
        card_head("Model-comparison table", "Same held-out test set · hyper-parameters and decision threshold tuned on validation data only · best value highlighted")
        ct = tab.drop(columns=["Test PR-AUC", "Test Precision", "Test Recall"]).copy()
        ct["Inference time (ms/1000)"] = ct["Inference time (ms/1000)"]
        gtable(ct, fmt={"Validation ROC-AUC": "{:.3f}", "Test ROC-AUC": "{:.3f}", "Test F1": "{:.3f}", "Test Brier": "{:.3f}", "Test Log-loss": "{:.3f}",
                        "Test ECE": "{:.3f}", "Training time (s)": "{:.2f}", "Inference time (ms/1000)": "{:.1f}"},
               best={"Test ROC-AUC": "max", "Validation ROC-AUC": "max", "Test F1": "max", "Test Brier": "min", "Test Log-loss": "min", "Test ECE": "min"},
               wrap=("Interpretability",))
        st.download_button("⬇ Download comparison table (CSV)", tab.to_csv(index=False).encode(), "model_comparison.csv", "text/csv")
    l, r = st.columns(2)
    colmap = {"Bayesian Model Averaging": VIOLET, "Best single BIC model": LILAC, "Logistic Regression": CYAN, "Naive Bayes": PINK, "Random Forest": AMBER}
    with l:
        with st.container(border=True):
            card_head("ROC curves", "")
            fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="rgba(255,255,255,.3)", dash="dash"), showlegend=False, hoverinfo="skip"))
            for nm, cl in colmap.items():
                fpr, tpr, _ = roc_curve(R["yte"], R["probs"][nm][1])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nm} ({roc_auc_score(R['yte'], R['probs'][nm][1]):.3f})", line=dict(color=cl, width=3 if nm.startswith("Bayes") else 1.8)))
            style_fig(fig, 420)
            fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)))
            fig.update_xaxes(title="false-positive rate")
            fig.update_yaxes(title="true-positive rate")
            plot(fig, "roc")
    with r:
        with st.container(border=True):
            card_head("Precision–recall curves", "Better for imbalanced defect data")
            fig = go.Figure()
            for nm, cl in colmap.items():
                pr, rc, _ = precision_recall_curve(R["yte"], R["probs"][nm][1])
                fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"{nm} ({average_precision_score(R['yte'], R['probs'][nm][1]):.3f})", line=dict(color=cl, width=3 if nm.startswith("Bayes") else 1.8)))
            style_fig(fig, 420)
            fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)))
            fig.update_xaxes(title="recall")
            fig.update_yaxes(title="precision")
            plot(fig, "pr")
    with st.container(border=True):
        card_head("Robustness: repeat the whole study on several random splits",
                  "A single split is noisy on small NASA datasets. This re-runs preprocessing, tuning and testing per seed.")
        nrep = st.slider("Number of random splits", 3, 10, 5)
        if st.button("▶ Run robustness study"):
            with st.spinner(f"Running {nrep} independent splits …"):
                st.session_state["rep"] = (data_key, nrep, get_repeated(data_key, buf.getvalue(), nrep, float(tau)))
        if "rep" in st.session_state and st.session_state["rep"][0] == data_key:
            rep = st.session_state["rep"][2]
            gtable(B.summarise_repeated(rep))
            fig = go.Figure()
            for nm, cl in colmap.items():
                d = rep[rep.Algorithm == nm]
                fig.add_trace(go.Box(y=d["Test ROC-AUC"], name=nm.replace("Bayesian Model Averaging", "BMA").replace("Logistic Regression", "LogReg").replace("Random Forest", "RF").replace("Best single BIC model", "Best-BIC"),
                                     marker_color=cl, line=dict(color=cl), boxmean=True, fillcolor="rgba(139,92,246,.12)"))
            style_fig(fig, 320, legend=False)
            fig.update_yaxes(title="Test ROC-AUC")
            plot(fig, "box")

# -----------------------------------------------------------------------------
# TAB 6 - PLAYGROUND
# -----------------------------------------------------------------------------
with tabs[5]:
    l, r = st.columns([1, 1.2])
    pipk = bma.inclusion_probabilities().head(6).index.tolist()
    Xtr_raw = sp["Xtr"]
    base = Xtr_raw.median()
    vals = {}
    with l:
        with st.container(border=True):
            card_head("Describe a module", "Sliders show the six most influential metrics · all others are set to the training median")
            for c in pipk:
                lo, hi = float(Xtr_raw[c].quantile(0.01)), float(Xtr_raw[c].quantile(0.995))
                if hi <= lo:
                    hi = lo + 1
                isint = bool(np.allclose(Xtr_raw[c].dropna() % 1, 0))
                if isint:
                    vals[c] = st.slider(c, int(lo), int(np.ceil(hi)), int(np.clip(base[c], lo, hi)))
                else:
                    vals[c] = st.slider(c, lo, hi, float(np.clip(base[c], lo, hi)), step=(hi - lo) / 200)
    x = base.copy()
    for k_, v_ in vals.items():
        x[k_] = v_
    Zx = pre.transform(pd.DataFrame([x]))
    dd = bma.predict_dist(Zx, level=level, n_draws=3000, seed=11)
    p, lo_, hi_ = dd["mean"][0], dd["lo"][0], dd["hi"][0]
    tier = "High" if lo_ >= thr else ("Low" if hi_ < thr else "Review")
    with r:
        with st.container(border=True):
            card_head("Live prediction", "Updates instantly as you move the sliders")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(color="#fff", size=44)),
                gauge=dict(axis=dict(range=[0, 100], tickcolor="#8f86bd"), bar=dict(color=TIER_COLOR[tier], thickness=.28), bgcolor="rgba(255,255,255,.05)",
                           borderwidth=0, steps=[dict(range=[lo_ * 100, hi_ * 100], color="rgba(167,139,250,.35)")],
                           threshold=dict(line=dict(color=AMBER, width=3), value=thr * 100))))
            style_fig(fig, 280, legend=False)
            fig.update_layout(margin=dict(l=40, r=40, t=40, b=10))
            plot(fig, "gauge")
            H(f"""<div style="text-align:center;margin-top:-6px">{pill(tier)} &nbsp; <span style="color:#b7aedf">{int(level*100)}% credible interval:
            <b style="color:#fff">{lo_*100:.0f}% – {hi_*100:.0f}%</b> &nbsp;·&nbsp; decision threshold {thr*100:.0f}%</span></div>""")
            fig = go.Figure(go.Histogram(x=dd["draws"][0], nbinsx=45, marker=dict(color=VIOLET, line=dict(width=0)), opacity=.9))
            fig.add_vline(x=thr, line_dash="dot", line_color=AMBER)
            style_fig(fig, 200, legend=False)
            fig.update_xaxes(range=[0, 1], title="posterior predictive of P(defect)")
            plot(fig, "playhist")
            st.caption(f"Epistemic uncertainty {dd['epistemic'][0]:.3f} nats · aleatoric {dd['aleatoric'][0]:.3f} nats. "
                       "A wide interval means the candidate models disagree here — treat the module as 'needs review'.")

# -----------------------------------------------------------------------------
# TAB 7 - ABOUT
# -----------------------------------------------------------------------------
with tabs[6]:
    l, r = st.columns(2)
    with l:
        with st.container(border=True):
            card_head("Method in one minute", "")
            st.markdown(f"""
**Goal.** Identify defect-prone software modules *and* quantify model uncertainty.

**1 · Preprocess (train only).** signed-log1p → drop constant → drop |ρ|>0.95 → z-score. Currently **{len(pre.columns_)}** metrics kept
({len(pre.dropped_correlated_)} correlated dropped).

**2 · Candidate models.** Bayesian logistic regressions over many metric subsets (greedy BIC forward selection + **MC3** search), Gaussian Naive Bayes. {bma.n_visited_} subsets visited, top {len(bma.models_)} kept.

**3 · Posterior model weights.** P(Mₖ|D) ∝ exp(−τ·BICₖ/2). Each logistic member also has a Laplace posterior over its coefficients.

**4 · Average predictions.** p(y=1|x) = Σₖ P(Mₖ|D)·E[p(y=1|x,θₖ,Mₖ)]; credible intervals come from sampling model → coefficients → probability.

**5 · Assess calibration.** Brier, log-loss, ECE, reliability diagrams.
""")
    with r:
        with st.container(border=True):
            card_head("Experimental protocol", "")
            st.markdown(f"""
- Exact duplicate modules removed first (they would leak across splits): **{n_all:,}** unique modules.
- Split **{int((1-test_size-0.2)*100)}/20/{int(test_size*100)}** train / validation / test, stratified, seed **{seed}**.
- **Same held-out test set** for every algorithm; preprocessing fitted on training only.
- Hyper-parameters (LogReg C, RF depth/leaf, NB smoothing) and the F1 decision threshold are tuned on the validation set only. BMA itself needs no validation data.
- Comparison algorithms: Logistic Regression, Random Forest, Naive Bayes (+ best single BIC model as an ablation).
- Metrics: ROC-AUC, F1, Brier score, log-loss (+ PR-AUC, ECE).
- NASA MDP data has no timestamps or entity IDs, so a stratified random split is used; this can be optimistic versus a real time-ordered deployment.
""")
    H('<div class="footer">DefectLens · Bayesian Model Averaging · NASA / PROMISE defect datasets · built with Streamlit &amp; Plotly</div>')
