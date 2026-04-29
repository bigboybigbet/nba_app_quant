"""
NBA AI Quant Pro — Streamlit entry point.
Run: streamlit run nba_app.py
All logic lives in engine/ and data/.  This file is UI only.
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from joblib import load

# ── internal modules ──────────────────────────────────────────────────────────
from config import (
    DATASET_FILE, ELO_FILE, FEATURE_COLS, HISTORY_FILE,
    MODEL_FILE, OPTUNA_PARAMS_FILE, PLAYER_IMPACT_FILE,
)
import data.cache as cache
from data.autobacktest import log_stats as ab_log_stats, run_auto_backtest
from data.elo import load_elo, save_elo, update_elo
from data.injuries import (
    apply_injury_adjustment, calc_strength_factor,
    get_injury_report, load_player_impact, load_player_impact_meta,
)
from data.odds import fetch_odds_api, fetch_odds_espn
from data.schedule import fail_reason, get_past_results_espn, get_schedule_by_date
from engine.betting import american_to_decimal, kelly_criterion
from engine.features import get_team_advanced_stats
from engine.models import build_master_dataset, predict_hybrid, train_model
from engine.simulate import run_monte_carlo, run_poisson_model

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna  # noqa: F401
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA AI Quant Pro", page_icon="🏀",
    layout="wide", initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&display=swap');

html, body { font-family: 'DM Sans', sans-serif; background: #0b0e1a !important; color: #e8eaf0; }
[data-testid="stApp"], [data-testid="stAppViewContainer"] { background: #0b0e1a !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMain"] > div { background: #0b0e1a !important; }
h1,h2,h3,h4,h5 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 1.5px; }

[data-testid="stSidebar"] { background: linear-gradient(160deg,#0f1629,#0b1020) !important; border-right: 1px solid #1e2d4a !important; }
[data-testid="stSidebar"] > div { background: transparent !important; }

[data-baseweb="tab"] { font-family: 'Bebas Neue', sans-serif !important; font-size: 1rem !important; letter-spacing: 1px !important; color: #667eea !important; background: transparent !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #f97316 !important; border-bottom: 2px solid #f97316 !important; }
[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1e2d4a !important; gap: 4px !important; }

.stButton > button,
button[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Bebas Neue', sans-serif !important; font-size: 1rem !important;
    letter-spacing: 1px !important; transition: all .2s ease !important;
}
button[data-testid="stBaseButton-primary"],
.stButton > button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg,#f97316,#ef4444) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Bebas Neue', sans-serif !important; font-size: 1rem !important;
    letter-spacing: 1px !important; transition: all .2s ease !important;
}
.stButton > button:hover, button[data-testid^="stBaseButton"]:hover {
    transform: translateY(-2px) !important; filter: brightness(1.12) !important;
    box-shadow: 0 8px 20px rgba(249,115,22,.35) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
    background: #111827 !important; border: 1px solid #1e3a5f !important;
    color: #e8eaf0 !important; border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #667eea !important; box-shadow: 0 0 0 2px rgba(102,126,234,.2) !important;
}
[data-testid="stTextInput"] label, [data-testid="stNumberInput"] label,
[data-testid="stDateInput"] label, [data-testid="stSelectbox"] label {
    color: #94a3b8 !important; font-size: .75rem !important;
    text-transform: uppercase !important; letter-spacing: .6px !important;
}
[data-testid="stNumberInput"] button { background: #1e2d4a !important; color: #94a3b8 !important; border: none !important; }

[data-testid="metric-container"],
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#111827,#1a2236) !important;
    border: 1px solid #1e2d4a !important; border-radius: 12px !important;
    padding: 16px 20px !important; box-shadow: 0 4px 20px rgba(0,0,0,.4) !important;
}
[data-testid="metric-container"] label,
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: .75rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="stMetricValue"] { font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important; color: #f97316 !important; }
[data-testid="stMetricDelta"] svg { display: none; }

[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
hr { border-color: #1e2d4a !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stProgress"] div[role="progressbar"],
.stProgress > div > div { background: linear-gradient(90deg,#667eea,#f97316) !important; }

.score-card { background: linear-gradient(135deg,#111827,#1a2236); border: 1px solid #1e3a5f; border-radius: 14px; padding: 24px; text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,.5); }
.team-name  { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; letter-spacing: 2px; }
.prob-big   { font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; color: #f97316; }
.vs-badge   { font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem; color: #667eea; }
.stat-row   { display: flex; justify-content: space-between; margin: 6px 0; font-size: .85rem; color: #94a3b8; }
.stat-val   { color: #e2e8f0; font-weight: 600; }
.injury-badge { background: #450a0a; color: #f87171; border-radius: 6px; padding: 2px 8px; font-size: .75rem; font-weight: 600; margin-left: 4px; }
.kelly-card { background: linear-gradient(135deg,#0a1f18,#0d1a14); border: 1px solid #065f46; border-radius: 14px; padding: 20px 24px; margin: 8px 0; }
.section-header { color: #94a3b8; font-size: .65rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [
    ("schedule_data", None), ("last_pick", {}), ("last_scan", None),
    ("home_odds_val", -110), ("away_odds_val", -110),
    ("odds_api_key", ""), ("odds_fetched_label", ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM HELPERS  (UI-only, stay here)
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(token, chat_id, msg):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=8,
        )
        return r.ok, r.json().get("description", "OK")
    except Exception as e:
        return False, str(e)


def format_picks_telegram(df, date_str):
    lines = [f"🏀 *NBA AI QUANT PRO — {date_str}*", ""]
    for _, row in df.iterrows():
        lines += [
            f"📌 *{row['Matchup']}*",
            f"  Pick: *{row['AI Pick']}* ({row['Confidence']})",
            f"  Spread: {row['Spread']}  |  O/U: {row['O/U']}",
            f"  Kelly bet: {row['Kelly%']} of bankroll", "",
        ]
    lines.append("_Generated by NBA AI Quant Pro_")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS  (matplotlib — presentational, stay here)
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG = "#0b0e1a"
CARD_BG = "#111827"
ORANGE  = "#f97316"
PURPLE  = "#667eea"
GREEN   = "#4ade80"
RED     = "#f87171"
GRAY    = "#374151"


def fig_mc_distributions(mc, ht, at):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(CARD_BG)
    axes[0].hist(mc["_h_sims"], bins=60, alpha=0.7, color=ORANGE, density=True, label=ht)
    axes[0].hist(mc["_a_sims"], bins=60, alpha=0.7, color=PURPLE, density=True, label=at)
    axes[0].axvline(np.mean(mc["_h_sims"]), color=ORANGE, ls="--", lw=1.5)
    axes[0].axvline(np.mean(mc["_a_sims"]), color=PURPLE, ls="--", lw=1.5)
    axes[0].set_title("Score Distribution", color="white", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, labelcolor="white", fontsize=8)
    axes[0].tick_params(colors="#94a3b8", labelsize=8)
    axes[0].set_xlabel("Points", color="#94a3b8", fontsize=8)
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    pos = mc["_margins"][mc["_margins"] > 0]
    neg = mc["_margins"][mc["_margins"] <= 0]
    axes[1].hist(pos, bins=60, alpha=0.8, color=ORANGE, density=True)
    axes[1].hist(neg, bins=60, alpha=0.8, color=PURPLE, density=True)
    axes[1].axvline(0, color="white", lw=1, ls="--")
    axes[1].axvline(mc["spread_margin"], color=ORANGE, lw=2)
    axes[1].set_title("Margin Distribution", color="white", fontsize=11, fontweight="bold")
    axes[1].tick_params(colors="#94a3b8", labelsize=8)
    axes[1].set_xlabel(f"← {at}  |  {ht} →", color="#94a3b8", fontsize=8)
    for sp in axes[1].spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=1.2)
    return fig


def fig_feature_importance(feat_imp):
    labels = list(feat_imp.keys())
    values = list(feat_imp.values())
    idx    = np.argsort(values)
    labels = [labels[i] for i in idx]
    values = [values[i] for i in idx]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    colors = [ORANGE if v == max(values) else PURPLE for v in values]
    ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)
    ax.set_title("Feature Importance", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_xlabel("Importance", color="#94a3b8", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=1.0)
    return fig


def fig_radar(h, a):
    cats  = ["eFG%", "TS%", "AST/TOV", "OREB%", "Reb", "+/-"]
    N     = len(cats)
    h_raw = [h["eFG_PCT"], h["TS_PCT"], min(h["AST_TOV"], 5) / 5,
              h["OREB_PCT"], h["REB"] / 50, (h["PLUS_MINUS"] + 15) / 30]
    a_raw = [a["eFG_PCT"], a["TS_PCT"], min(a["AST_TOV"], 5) / 5,
              a["OREB_PCT"], a["REB"] / 50, (a["PLUS_MINUS"] + 15) / 30]
    h_raw = [max(0, min(1, v)) for v in h_raw]
    a_raw = [max(0, min(1, v)) for v in a_raw]
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    h_vals  = h_raw + [h_raw[0]]
    a_vals  = a_raw + [a_raw[0]]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot(angles, h_vals, color=ORANGE, lw=2)
    ax.fill(angles, h_vals, color=ORANGE, alpha=0.25)
    ax.plot(angles, a_vals, color=PURPLE, lw=2)
    ax.fill(angles, a_vals, color=PURPLE, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color="#94a3b8", size=8)
    ax.tick_params(colors="#94a3b8")
    ax.yaxis.set_visible(False)
    ax.grid(color=GRAY, lw=0.5)
    ax.spines["polar"].set_color(GRAY)
    hp = mpatches.Patch(color=ORANGE, label=h["name"].split()[-1])
    ap = mpatches.Patch(color=PURPLE, label=a["name"].split()[-1])
    ax.legend(handles=[hp, ap], loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, labelcolor="white", fontsize=8)
    fig.tight_layout()
    return fig


def fig_shap_chart(model, hd, ad):
    if not SHAP_AVAILABLE:
        return None
    try:
        def _row(s, is_home):
            return pd.DataFrame([[
                is_home, s["PLUS_MINUS"], s["PTS"], s["REB"],
                s["eFG_PCT"], s["AST_TOV"], s["TS_PCT"],
                s["WIN_STREAK"], s["OREB_PCT"], s["DAYS_REST"], s["IS_B2B"],
            ]], columns=FEATURE_COLS)
        explainer = shap.TreeExplainer(model)
        h_sv = explainer.shap_values(_row(hd, 1))[0]
        a_sv = explainer.shap_values(_row(ad, 0))[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        fig.patch.set_facecolor(DARK_BG)
        for ax, sv, title in zip(axes, [h_sv, a_sv],
                                  [hd["name"].split()[-1], ad["name"].split()[-1]]):
            ax.set_facecolor(CARD_BG)
            cols = [ORANGE if v > 0 else RED for v in sv]
            ax.barh(FEATURE_COLS, sv, color=cols, edgecolor="none", height=0.6)
            ax.axvline(0, color="white", lw=0.8, ls="--")
            ax.set_title(f"SHAP — {title}", color="white", fontsize=10, fontweight="bold")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.tight_layout(pad=1.0)
        return fig
    except Exception:
        return None


def fig_calibration(val_proba, val_y):
    proba = np.array(val_proba)
    y     = np.array(val_y)
    bins  = np.linspace(0, 1, 11)
    means, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() > 0:
            means.append(proba[mask].mean())
            accs.append(y[mask].mean())
            counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="Perfect")
    sc = ax.scatter(means, accs, c=counts, cmap="YlOrRd", s=80, zorder=3)
    ax.plot(means, accs, color=ORANGE, lw=2)
    plt.colorbar(sc, ax=ax, label="Count").ax.yaxis.label.set_color("#94a3b8")
    ax.set_xlabel("Predicted prob", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Actual win rate", color="#94a3b8", fontsize=9)
    ax.set_title("Calibration Curve", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.legend(frameon=False, labelcolor="white", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return fig


def fig_roi_chart(hist):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(CARD_BG)
    cum = hist["PnL"].cumsum()
    col = ORANGE if cum.iloc[-1] >= 0 else RED
    axes[0].plot(range(len(cum)), cum, color=col, lw=2)
    axes[0].axhline(0, color=GRAY, ls="--", lw=1)
    axes[0].fill_between(range(len(cum)), cum, 0, alpha=0.15, color=col)
    axes[0].set_title("Cumulative P&L (units)", color="white", fontsize=11, fontweight="bold")
    axes[0].tick_params(colors="#94a3b8", labelsize=8)
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    win_roll = hist["Correct"].rolling(20, min_periods=5).mean() * 100
    axes[1].plot(range(len(win_roll)), win_roll, color=PURPLE, lw=2)
    axes[1].axhline(52.4, color=ORANGE, ls="--", lw=1, label="Break-even ~52.4%")
    axes[1].set_title("Rolling Win Rate (20 games)", color="white", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("%", color="#94a3b8", fontsize=8)
    axes[1].tick_params(colors="#94a3b8", labelsize=8)
    axes[1].legend(frameon=False, labelcolor="white", fontsize=8)
    for sp in axes[1].spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=1.2)
    return fig


def fig_poisson_bars(pr, ht, at):
    qs  = [f"Q{q['Q']}" for q in pr["quarters"]]
    hs  = [q["Home"] for q in pr["quarters"]]
    as_ = [q["Away"] for q in pr["quarters"]]
    x, w = np.arange(4), 0.35
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.bar(x - w / 2, hs, w, label=ht, color=ORANGE, alpha=0.85)
    ax.bar(x + w / 2, as_, w, label=at, color=PURPLE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(qs, color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_title("Projected Quarter Scoring (Poisson)", color="white", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, labelcolor="white", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 16px'>
        <div style='font-family:Bebas Neue;font-size:2rem;color:#f97316;letter-spacing:3px'>NBA AI QUANT</div>
        <div style='font-size:.65rem;color:#667eea;letter-spacing:2px'>PRO v4 · MODULAR</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── Model control ─────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Model Control")
    force_rebuild  = st.toggle("Force full rebuild", value=False)
    use_optuna_tog = st.toggle("Optuna HPO", value=False, disabled=not OPTUNA_AVAILABLE,
                               help="Bayesian HPO — runs optuna_tuner.py. pip install optuna")
    if st.button("🔄 Train / Reload Model", use_container_width=True):
        msgs = []
        with st.spinner("Training model…"):
            model, count, metrics = train_model(
                force_rebuild, use_optuna_tog,
                status_cb=lambda m: msgs.append(m),
            )
        for m in msgs:
            st.caption(m)
        if model:
            st.session_state.update({
                "xgb_model": model, "model_metrics": metrics,
                "feat_imp": metrics["feat_imp"],
            })
            st.success(f"✅ Model ready — {count:,} samples")
            st.caption(
                f"Val Acc: **{metrics['val_acc']:.1%}** | "
                f"Loss: **{metrics['val_loss']:.4f}** | "
                f"Iter: **{metrics['best_iter']}**"
            )
        else:
            st.error("Training failed — check NBA API connection.")
    st.divider()

    # ── Auto-backtest ─────────────────────────────────────────────────────
    st.markdown("#### ⚡ Smart Auto-Backtest")
    _ab = ab_log_stats()
    st.caption(f"📋 {_ab['total']:,} games backtested so far")
    if st.button("🔍 Backtest New Games Only", use_container_width=True,
                 help="Detects unprocessed completed games, blindfold-predicts them using pre-game data, retrains on mistakes"):
        if not os.path.exists(MODEL_FILE):
            st.error("⚠️ Train the model first.")
        else:
            _ab_model = load(MODEL_FILE)
            _ab_msgs  = []
            with st.spinner("Scanning for new games…"):
                _ab_result = run_auto_backtest(
                    _ab_model, status_cb=lambda m: _ab_msgs.append(m)
                )
            for m in _ab_msgs:
                st.caption(m)
            if _ab_result["new"] == 0:
                st.info("✅ All games already backtested — nothing new.")
            else:
                st.success(
                    f"✅ {_ab_result['new']} new games · "
                    f"{_ab_result['correct']}/{_ab_result['new']} correct "
                    f"({_ab_result['acc'] * 100:.1f}%) · "
                    f"{_ab_result['lessons']} lessons added"
                )
                if _ab_result["metrics"]:
                    m = _ab_result["metrics"]
                    st.caption(
                        f"Retrained — Val Acc: **{m['val_acc']:.1%}** | "
                        f"Loss: **{m['val_loss']:.4f}**"
                    )
                    st.session_state.update({
                        "xgb_model": _ab_model, "model_metrics": m,
                        "feat_imp": m["feat_imp"],
                    })
    st.divider()

    # ── Feature importance ────────────────────────────────────────────────
    if "feat_imp" in st.session_state:
        st.markdown("#### 📊 Feature Importance")
        st.pyplot(fig_feature_importance(st.session_state["feat_imp"]),
                  use_container_width=True)
        st.divider()

    # ── Calibration ───────────────────────────────────────────────────────
    if "model_metrics" in st.session_state and \
       "val_proba" in st.session_state.get("model_metrics", {}):
        m = st.session_state["model_metrics"]
        st.markdown("#### 📐 Calibration")
        st.pyplot(fig_calibration(m["val_proba"], m["val_y"]),
                  use_container_width=True)
        st.divider()

    # ── Player impact ─────────────────────────────────────────────────────
    st.markdown("#### 🩹 Injury Impact Data")
    _meta = load_player_impact_meta()
    if _meta["count"] > 0:
        st.caption(f"✅ {_meta['count']} players · updated {_meta['updated']}")
    else:
        st.caption("⚠️ Using fallback (20 players). Run updater for full coverage.")

    if st.button("📥 Update Player Impact", use_container_width=True,
                 help="Fetches current season stats — ~10 sec"):
        _script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "player_impact_updater.py")
        with st.spinner("Fetching NBA player stats…"):
            proc = subprocess.run([sys.executable, _script],
                                  capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            st.success("✅ Player impact updated!")
        else:
            st.error("Update failed — check NBA API connection.")
    st.divider()

    # ── Cache stats ───────────────────────────────────────────────────────
    st.markdown("#### 💾 API Cache")
    _cs = cache.stats()
    st.caption(f"Live: {_cs['live']} entries · Expired: {_cs['expired']}")
    if st.button("🗑️ Clear Cache", use_container_width=True):
        cache.clear_all()
        st.success("Cache cleared.")
    st.divider()

    st.caption("NBA Stats API + ESPN · 2 Seasons · 50k Sims · Auto-Backtest")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER + TABS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:8px 0 12px;border-bottom:1px solid #1e2d4a;margin-bottom:4px'>
  <span style='font-family:Bebas Neue,sans-serif;font-size:2.2rem;letter-spacing:3px;color:#f1f5f9'>
    🏀 NBA HYBRID QUANT PRO
  </span>
  <span style='font-size:.72rem;color:#667eea;margin-left:14px;letter-spacing:2px;vertical-align:middle'>
    XGBOOST · MONTE CARLO · POISSON · ELO · KELLY · INJURY
  </span>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  SINGLE GAME", "💰  RADAR SCANNER",
    "🔬  BACKTEST",    "📈  ROI DASHBOARD", "⚙️  SETTINGS",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE GAME
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        t1, t2 = st.columns(2)
        ht = t1.text_input("🏠 Home Team", "Lakers", key="ht_input")
        at = t2.text_input("✈️ Away Team", "Warriors", key="at_input")

        odds_hdr, fetch_col = st.columns([3, 1])
        odds_hdr.markdown("**Moneyline Odds**")
        fetch_btn = fetch_col.button("🔄 Auto-fetch", use_container_width=True,
                                     help="Pull live moneylines from ESPN / Odds API")
        if fetch_btn:
            with st.spinner("Fetching live moneylines…"):
                _key   = st.session_state.get("odds_api_key", "")
                result = fetch_odds_api(ht, at, _key) if _key else None
                if not result:
                    result = fetch_odds_espn(ht, at)
            if isinstance(result, tuple) and result[0] is not None:
                ho, ao, src = result
                st.session_state["home_odds_val"]     = ho
                st.session_state["away_odds_val"]     = ao
                st.session_state["odds_fetched_label"] = f"📡 {src} · Home {ho:+d} / Away {ao:+d}"
            elif isinstance(result, tuple) and result[2] == "game_found":
                st.session_state["odds_fetched_label"] = ""
                st.warning("⚠️ Game found but no odds published yet. Try closer to tip-off.")
            else:
                st.session_state["odds_fetched_label"] = ""
                st.warning("⚠️ Game not on today's schedule. Check team names or add an Odds API key in ⚙️ Settings.")

        if st.session_state["odds_fetched_label"]:
            st.caption(st.session_state["odds_fetched_label"])

        col_o1, col_o2 = st.columns(2)
        home_odds = col_o1.number_input("Home", min_value=-10000, max_value=10000,
                                        step=1, key="home_odds_val")
        away_odds = col_o2.number_input("Away", min_value=-10000, max_value=10000,
                                        step=1, key="away_odds_val")
        st.write("")
        run_btn = st.button("🚀 RUN FULL ANALYSIS", type="primary", use_container_width=True)

    with right:
        st.markdown("""
        <div class='score-card' style='margin-top:8px'>
            <div style='color:#94a3b8;font-size:.7rem;letter-spacing:2px;margin-bottom:8px'>ENGINE STACK v4</div>
            <div style='text-align:left;font-size:.82rem;color:#cbd5e1;line-height:1.9'>
                🧠 <b>XGBoost</b> — 11 features + Optuna HPO<br>
                🎲 <b>Monte Carlo</b> — 50,000 simulations<br>
                🎯 <b>Poisson</b> — Quarter-by-quarter model<br>
                🏅 <b>Elo ratings</b> — Dynamic team power<br>
                🩹 <b>Injury system</b> — Live ESPN + impact data<br>
                💰 <b>Kelly Criterion</b> — Optimal bet sizing<br>
                🔍 <b>SHAP</b> — Model explainability<br>
                💾 <b>SQLite cache</b> — Fast repeated lookups
            </div>
        </div>""", unsafe_allow_html=True)

    if run_btn:
        if not os.path.exists(MODEL_FILE):
            st.error("⚠️ No model found. Click **Train / Reload Model** in the sidebar first.")
        else:
            model = load(MODEL_FILE)
            with st.spinner("⚡ Fetching live stats & injuries, running 50k simulations…"):
                hd    = get_team_advanced_stats(ht)
                ad    = get_team_advanced_stats(at)
                h_inj = get_injury_report(ht) if hd else []
                a_inj = get_injury_report(at) if ad else []

            if not hd or not ad:
                st.error("❌ Team not found. Try full name e.g. 'Los Angeles Lakers'.")
            else:
                fp, spread, total, xgb_p, mc_p, conflict, mc_full, hf, af = \
                    predict_hybrid(hd, ad, model, h_inj, a_inj)
                conf   = max(fp, 1 - fp)
                winner = hd["name"] if fp > 0.5 else ad["name"]
                pr     = run_poisson_model(hd, ad)
                h_full_k, h_frac_k = kelly_criterion(fp,       american_to_decimal(home_odds))
                a_full_k, a_frac_k = kelly_criterion(1 - fp,   american_to_decimal(away_odds))

                # Injury report
                if h_inj or a_inj:
                    with st.expander("🩹 Injury Report (live ESPN)", expanded=True):
                        ic1, ic2 = st.columns(2)
                        with ic1:
                            st.markdown(f"**{hd['name']}** — strength `{hf:.0%}`")
                            for p in h_inj:
                                st.markdown(
                                    f"- {p['name']} <span class='injury-badge'>{p['status'].upper()}</span> {p['detail']}",
                                    unsafe_allow_html=True)
                            if not h_inj:
                                st.markdown("✅ No key injuries")
                        with ic2:
                            st.markdown(f"**{ad['name']}** — strength `{af:.0%}`")
                            for p in a_inj:
                                st.markdown(
                                    f"- {p['name']} <span class='injury-badge'>{p['status'].upper()}</span> {p['detail']}",
                                    unsafe_allow_html=True)
                            if not a_inj:
                                st.markdown("✅ No key injuries")

                # Consensus banner
                if conflict:
                    st.markdown("""<div style='background:#450a0a;border:1px solid #b91c1c;border-radius:10px;padding:14px 20px;margin-bottom:16px'>
                        🚨 <b>MODEL CONFLICT</b> — XGBoost and Monte Carlo disagree. Proceed with caution.</div>""",
                                unsafe_allow_html=True)
                else:
                    st.markdown("""<div style='background:#14532d;border:1px solid #16a34a;border-radius:10px;padding:14px 20px;margin-bottom:16px'>
                        ✅ <b>HIGH CONSENSUS</b> — All engines agree on the same winner.</div>""",
                                unsafe_allow_html=True)

                # Engine metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🧠 XGBoost",    f"{xgb_p * 100:.1f}%")
                c2.metric("🎲 Monte Carlo", f"{mc_p * 100:.1f}%")
                c3.metric("🎯 Poisson",     f"{pr['poisson_h_prob'] * 100:.1f}%")
                c4.metric("⚖️ Hybrid",      f"{fp * 100:.1f}%",
                          delta="✅ SAFE" if conf >= 0.65 and not conflict else "⚠️ RISKY")
                st.divider()

                # Team cards
                ra, rb, rc = st.columns([1.1, 0.6, 1.1], gap="small")
                with ra:
                    bh    = int(fp * 100)
                    b2b_h = "<span class='injury-badge'>B2B</span>" if hd["IS_B2B"] else ""
                    st.markdown(f"""
                    <div class='score-card'>
                        <div class='team-name' style='color:#f97316'>{hd['name'].split()[-1].upper()} {b2b_h}</div>
                        <div style='color:#94a3b8;font-size:.7rem;margin:4px 0 10px'>HOME · ELO {hd['ELO']:.0f}</div>
                        <div class='prob-big'>{fp * 100:.0f}%</div>
                        <div style='background:#1e2d4a;border-radius:6px;height:8px;margin:12px 0'>
                            <div style='background:#f97316;width:{bh}%;height:8px;border-radius:6px'></div>
                        </div>
                        <div class='stat-row'><span>eFG%</span><span class='stat-val'>{hd['eFG_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>TS%</span><span class='stat-val'>{hd['TS_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>AST/TOV</span><span class='stat-val'>{hd['AST_TOV']:.2f}</span></div>
                        <div class='stat-row'><span>Win Streak</span><span class='stat-val'>{int(hd['WIN_STREAK'])}</span></div>
                        <div class='stat-row'><span>Days Rest</span><span class='stat-val'>{int(hd['DAYS_REST'])}</span></div>
                        <div class='stat-row'><span>Avg PTS</span><span class='stat-val'>{hd['PTS']:.1f}</span></div>
                        <div class='stat-row'><span>Team Strength</span><span class='stat-val'>{hf:.0%}</span></div>
                    </div>""", unsafe_allow_html=True)
                with rb:
                    st.markdown("""<div style='display:flex;align-items:center;justify-content:center;height:100%;padding-top:60px'>
                        <div class='vs-badge'>VS</div></div>""", unsafe_allow_html=True)
                with rc:
                    ba    = int((1 - fp) * 100)
                    b2b_a = "<span class='injury-badge'>B2B</span>" if ad["IS_B2B"] else ""
                    st.markdown(f"""
                    <div class='score-card'>
                        <div class='team-name' style='color:#667eea'>{ad['name'].split()[-1].upper()} {b2b_a}</div>
                        <div style='color:#94a3b8;font-size:.7rem;margin:4px 0 10px'>AWAY · ELO {ad['ELO']:.0f}</div>
                        <div class='prob-big' style='color:#667eea'>{(1 - fp) * 100:.0f}%</div>
                        <div style='background:#1e2d4a;border-radius:6px;height:8px;margin:12px 0'>
                            <div style='background:#667eea;width:{ba}%;height:8px;border-radius:6px'></div>
                        </div>
                        <div class='stat-row'><span>eFG%</span><span class='stat-val'>{ad['eFG_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>TS%</span><span class='stat-val'>{ad['TS_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>AST/TOV</span><span class='stat-val'>{ad['AST_TOV']:.2f}</span></div>
                        <div class='stat-row'><span>Win Streak</span><span class='stat-val'>{int(ad['WIN_STREAK'])}</span></div>
                        <div class='stat-row'><span>Days Rest</span><span class='stat-val'>{int(ad['DAYS_REST'])}</span></div>
                        <div class='stat-row'><span>Avg PTS</span><span class='stat-val'>{ad['PTS']:.1f}</span></div>
                        <div class='stat-row'><span>Team Strength</span><span class='stat-val'>{af:.0%}</span></div>
                    </div>""", unsafe_allow_html=True)

                st.divider()

                # Monte Carlo
                st.markdown("#### 🎲 Monte Carlo  *(50,000 sims)*")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📐 Spread",
                          f"{hd['name'].split()[-1]} -{spread:.1f}" if spread > 0
                          else f"{ad['name'].split()[-1]} -{abs(spread):.1f}",
                          help=f"IQR: {mc_full['spread_lo']:.1f}–{mc_full['spread_hi']:.1f}")
                m2.metric("🔢 O/U",      f"{total:.1f}",
                          help=f"IQR: {mc_full['total_lo']:.1f}–{mc_full['total_hi']:.1f}")
                m3.metric("🏠 Home %",   f"{mc_full['mc_home_prob'] * 100:.1f}%")
                m4.metric("⏱️ OT %",     f"{mc_full['ot_prob'] * 100:.1f}%")

                # Poisson
                st.markdown("#### 🎯 Poisson Quarter Projection")
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Proj. Score",    f"{pr['proj_h']} – {pr['proj_a']}")
                pc2.metric("Median Total",   f"{pr['median_total']:.1f}")
                pc3.metric("Over (80th %)",  f"{pr['ou_80']:.1f}")
                pc4.metric("Under (20th %)", f"{pr['ou_20']:.1f}")

                # Kelly
                _hn = hd["name"].split()[-1]
                _an = ad["name"].split()[-1]
                st.markdown(f"""
<div class='kelly-card'>
  <div class='section-header'>💰 Kelly Criterion — Optimal Bet Sizing</div>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;text-align:center'>
    <div>
      <div style='color:#94a3b8;font-size:.7rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>🏠 {_hn}</div>
      <div style='font-family:Bebas Neue,sans-serif;font-size:3rem;color:#4ade80;line-height:1'>{h_frac_k:.1f}<span style='font-size:1.1rem'>%</span></div>
      <div style='color:#667eea;font-size:.78rem;margin-top:4px'>Full Kelly: {h_full_k:.1f}%</div>
    </div>
    <div>
      <div style='color:#94a3b8;font-size:.7rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>✈️ {_an}</div>
      <div style='font-family:Bebas Neue,sans-serif;font-size:3rem;color:#4ade80;line-height:1'>{a_frac_k:.1f}<span style='font-size:1.1rem'>%</span></div>
      <div style='color:#667eea;font-size:.78rem;margin-top:4px'>Full Kelly: {a_full_k:.1f}%</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                # Charts
                ch1, ch2 = st.columns([2.2, 1])
                with ch1:
                    st.pyplot(fig_mc_distributions(mc_full, _hn, _an),
                              use_container_width=True)
                with ch2:
                    st.pyplot(fig_radar(hd, ad), use_container_width=True)
                st.pyplot(fig_poisson_bars(pr, _hn, _an), use_container_width=True)

                if SHAP_AVAILABLE:
                    st.markdown("#### 🔍 SHAP — Why this pick?")
                    sf = fig_shap_chart(model, hd, ad)
                    if sf:
                        st.pyplot(sf, use_container_width=True)
                else:
                    st.info("💡 `pip install shap` for explainability charts")

                # Final pick
                if not conflict and conf >= 0.55:
                    kelly_show = h_frac_k if fp > 0.5 else a_frac_k
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#1a2236,#111827);border:1px solid #f97316;
                                border-radius:12px;padding:20px 24px;margin-top:16px;text-align:center'>
                        <div style='color:#94a3b8;font-size:.7rem;letter-spacing:2px;margin-bottom:6px'>AI PICK</div>
                        <div style='font-family:Bebas Neue;font-size:2rem;color:#f97316'>{winner}</div>
                        <div style='color:#94a3b8;font-size:.85rem'>
                            to win · {conf * 100:.1f}% confidence · Kelly bet: {kelly_show:.1f}% of bankroll
                        </div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RADAR SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    sc1, sc2 = st.columns([1, 1], gap="large")
    with sc1:
        target_date = st.date_input("🗓️ Game date", datetime.today())
    with sc2:
        with st.expander("📱 Telegram — auto-send picks"):
            tg_token = st.text_input("Bot Token", type="password", key="tg2",
                                     placeholder="From @BotFather")
            tg_chat  = st.text_input("Chat ID", key="tg2_chat",
                                     placeholder="From @userinfobot")
    scan_btn = st.button("🔍 Scan All Games & Calculate Lines",
                         type="primary", use_container_width=True)

    if scan_btn:
        if not os.path.exists(MODEL_FILE):
            st.error("⚠️ Train the model first.")
        else:
            df_sched = get_schedule_by_date(target_date)
            if df_sched is None or df_sched.empty:
                st.warning("No games found for this date.")
            else:
                model = load(MODEL_FILE)
                recs  = []
                prog  = st.progress(0)
                n     = len(df_sched)
                for i, row in df_sched.iterrows():
                    hn, an = row["Home Team"], row["Away Team"]
                    hd = get_team_advanced_stats(hn)
                    ad = get_team_advanced_stats(an)
                    if hd and ad:
                        hi = get_injury_report(hn)
                        ai = get_injury_report(an)
                        fp, spread, total, xp, mp, conf_f, mc_full, hf, af = \
                            predict_hybrid(hd, ad, model, hi, ai)
                        conf = max(fp, 1 - fp)
                        pick = hd["name"] if fp > 0.5 else ad["name"]
                        _, fk = kelly_criterion(fp if fp > 0.5 else 1 - fp, 1.91)
                        recs.append({
                            "Matchup":    f"{hn} vs {an}",
                            "AI Pick":    pick,
                            "Confidence": f"{conf * 100:.1f}%",
                            "XGBoost":    f"{xp * 100:.1f}%",
                            "MC":         f"{mp * 100:.1f}%",
                            "Spread":     f"{hn.split()[-1]} -{spread:.1f}" if spread > 0
                                          else f"{an.split()[-1]} -{abs(spread):.1f}",
                            "O/U":        f"{total:.1f}",
                            "OT%":        f"{mc_full['ot_prob'] * 100:.1f}%",
                            "H Str.":     f"{hf:.0%}",
                            "A Str.":     f"{af:.0%}",
                            "Kelly%":     f"{fk:.1f}%",
                            "Status":     "✅ Consensus" if not conf_f else "🚨 Conflict",
                        })
                    prog.progress((i + 1) / n)

                if recs:
                    result_df = pd.DataFrame(recs)
                    st.dataframe(result_df, hide_index=True, use_container_width=True)
                    st.session_state["last_scan"] = result_df
                    vip = result_df[
                        (result_df["Status"] == "✅ Consensus") &
                        (result_df["Confidence"].str.rstrip("%").astype(float) >= 65)
                    ]
                    if not vip.empty:
                        st.markdown("#### 🔥 VIP Picks (≥65% + consensus)")
                        st.dataframe(vip, hide_index=True, use_container_width=True)
                        if tg_token and tg_chat:
                            if st.button("📤 Send VIP Picks to Telegram"):
                                msg = format_picks_telegram(vip, target_date.strftime("%Y-%m-%d"))
                                ok, desc = send_telegram(tg_token, tg_chat, msg)
                                st.success("✅ Sent!") if ok else st.error(f"❌ {desc}")
                else:
                    st.warning("Could not compute predictions.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.info("Replay historical dates to measure accuracy. Wrong predictions are re-weighted and the model self-updates.")
    bt_col, _ = st.columns([1.2, 1])
    with bt_col:
        date_range = st.date_input(
            "🗓️ Backtest range",
            [datetime.today() - timedelta(days=3), datetime.today() - timedelta(days=1)],
        )
    bt_btn = st.button("🚀 RUN BACKTEST & SELF-LEARN", type="primary", use_container_width=True)

    if bt_btn:
        if len(date_range) != 2:
            st.error("Select valid start and end dates.")
        elif not os.path.exists(MODEL_FILE):
            st.error("⚠️ Train model first.")
        else:
            model = load(MODEL_FILE)
            sd, ed = date_range
            all_eval, new_lessons = [], []
            tg = tc = hcg = hcc = 0
            pb         = st.progress(0)
            days_total = (ed - sd).days + 1
            cd, dc     = sd, 0

            with st.spinner("🕰️ Running time-machine backtest…"):
                while cd <= ed:
                    for game in get_past_results_espn(cd):
                        hn, an, act = game["Home Team"], game["Away Team"], game["Actual_Winner"]
                        hd = get_team_advanced_stats(hn, target_date=cd)
                        ad = get_team_advanced_stats(an, target_date=cd)
                        if hd and ad:
                            fp, spread, total, xp, mp, cfl, mc_full, hf, af = \
                                predict_hybrid(hd, ad, model)
                            pred = 1 if fp > 0.5 else 0
                            conf = max(fp, 1 - fp)
                            ok   = pred == act
                            tg  += 1
                            if ok:
                                tc += 1
                            if conf >= 0.65 and not cfl:
                                hcg += 1
                                if ok:
                                    hcc += 1
                            if not ok:
                                for ih, stats, res in [(1, hd, act), (0, ad, 1 - act)]:
                                    new_lessons.append({
                                        "GAME_DATE": cd.strftime("%Y-%m-%d"), "TEAM_ID": 0,
                                        "IS_HOME": ih, "PLUS_MINUS": stats["PLUS_MINUS"],
                                        "PTS": stats["PTS"], "REB": stats["REB"],
                                        "eFG_PCT": stats["eFG_PCT"], "AST_TOV": stats["AST_TOV"],
                                        "TS_PCT": stats["TS_PCT"], "WIN_STREAK": stats["WIN_STREAK"],
                                        "OREB_PCT": stats["OREB_PCT"], "DAYS_REST": stats["DAYS_REST"],
                                        "IS_B2B": stats["IS_B2B"], "RESULT": res,
                                    })
                            elo_d = load_elo()
                            w = hn if act == 1 else an
                            l = an if act == 1 else hn
                            save_elo(update_elo(w, l, elo_d))
                            all_eval.append({
                                "Date":    cd.strftime("%d/%m"),
                                "Matchup": f"{hn} vs {an}",
                                "Score":   f"{game['H_Score']}–{game['A_Score']}",
                                "AI Pick": f"🏠 Home ({conf * 100:.1f}%)" if pred == 1
                                           else f"✈️ Away ({conf * 100:.1f}%)",
                                "Engines": "🚨 Conflict" if cfl else "✅ Agree",
                                "Result":  "✅ Win" if ok else "❌ Loss",
                                "Reason":  "" if ok else fail_reason(game, pred, conf),
                            })
                    dc += 1
                    pb.progress(dc / days_total)
                    cd += timedelta(days=1)

            if tg > 0:
                st.markdown("### 📊 Backtest Performance Report")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Games",       f"{tg}")
                r2.metric("Overall Acc.", f"{tc / tg * 100:.1f}%", delta=f"+{tc} correct")
                r3.metric("VIP Acc.",    f"{hcc / hcg * 100:.1f}%" if hcg > 0 else "N/A",
                          delta=f"{hcg} VIP" if hcg > 0 else "")
                r4.metric("Wrong",       f"{tg - tc}", delta=f"{len(new_lessons) // 2} lessons")
                st.dataframe(pd.DataFrame(all_eval), hide_index=True, use_container_width=True)

                if new_lessons and os.path.exists(DATASET_FILE):
                    st.markdown("---")
                    st.warning(f"⚠️ {len(new_lessons) // 2} wrong predictions → retraining…")
                    master  = pd.read_csv(DATASET_FILE)
                    lessons = pd.DataFrame(new_lessons)
                    updated = pd.concat([master, lessons, lessons, lessons], ignore_index=True)
                    updated.to_csv(DATASET_FILE, index=False)
                    with st.spinner("🧬 Self-learning…"):
                        _, cnt, nm = train_model(force_rebuild=False)
                    if nm:
                        st.success(f"🧬 EVOLUTION COMPLETE — {cnt:,} samples · Val acc: **{nm['val_acc']:.1%}**")
            else:
                st.warning("No completed results found in this date range.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📈 Prediction History & ROI Tracker")
    st.info("Log your actual game results here to track bankroll P&L using Kelly-sized bets.")

    with st.expander("➕ Log a completed bet"):
        lc1, lc2, lc3, lc4 = st.columns(4)
        log_m = lc1.text_input("Matchup",    "Lakers vs Warriors")
        log_p = lc2.text_input("Your pick",  "Lakers")
        log_c = lc3.number_input("Confidence %", 50.0, 100.0, 65.0, step=0.5)
        log_o = lc4.number_input("Odds (American)", -500, 500, -110)
        log_s = st.number_input("Stake (Kelly % of bankroll used)", 0.0, 10.0, 2.0, step=0.5)
        log_r = st.selectbox("Result", ["Win", "Loss"])
        if st.button("💾 Save to History", use_container_width=True):
            dec = american_to_decimal(log_o)
            pnl = log_s * (dec - 1) if log_r == "Win" else -log_s
            row = {
                "Date": datetime.today().strftime("%Y-%m-%d"), "Matchup": log_m,
                "Pick": log_p, "Confidence": log_c, "Odds": log_o,
                "Stake%": log_s, "Result": log_r,
                "Correct": 1 if log_r == "Win" else 0, "PnL": round(pnl, 2),
            }
            hist = pd.concat(
                [pd.read_csv(HISTORY_FILE), pd.DataFrame([row])], ignore_index=True
            ) if os.path.exists(HISTORY_FILE) else pd.DataFrame([row])
            hist.to_csv(HISTORY_FILE, index=False)
            st.success("✅ Logged!")

    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        if not hist.empty:
            total_pnl = hist["PnL"].sum()
            wr  = hist["Correct"].mean() * 100
            roi = (total_pnl / hist["Stake%"].sum()) * 100 if hist["Stake%"].sum() > 0 else 0
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Bets", f"{len(hist)}")
            m2.metric("Win Rate",   f"{wr:.1f}%")
            m3.metric("Total P&L",  f"{total_pnl:+.2f} units",
                      delta="Profitable" if total_pnl > 0 else "Loss")
            m4.metric("ROI",        f"{roi:+.1f}%")
            st.pyplot(fig_roi_chart(hist), use_container_width=True)
            st.dataframe(hist.sort_values("Date", ascending=False),
                         hide_index=True, use_container_width=True)
    else:
        st.info("No history yet. Log your first bet above.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### 📊 Odds API (auto-fetch moneylines)")
    st.markdown("""
    Get a **free** key at [the-odds-api.com](https://the-odds-api.com) (500 req/month).
    Leave blank to use ESPN odds only (today's games, no key required).
    """)
    st.text_input("The Odds API Key", type="password", key="odds_api_key",
                  placeholder="Paste your key — stored in session only, never saved to disk")
    st.divider()

    st.markdown("#### 📱 Telegram Bot Setup")
    st.markdown("""
    1. Message **@BotFather** on Telegram → `/newbot` → copy your **Bot Token**
    2. Message **@userinfobot** → copy your **Chat ID**
    3. Enter both in the Radar Scanner tab to auto-send VIP picks
    """)
    with st.expander("🔧 Test connection"):
        tt = st.text_input("Bot Token", type="password", key="tg_test")
        tc = st.text_input("Chat ID",   key="tg_test_chat")
        if st.button("📤 Send test message"):
            if tt and tc:
                ok, desc = send_telegram(tt, tc, "🏀 NBA AI Quant Pro — connection test ✅")
                st.success("Connected!") if ok else st.error(f"Error: {desc}")
            else:
                st.warning("Enter both fields.")
    st.divider()

    st.markdown("#### 📦 Installed Packages")
    pkgs = {
        "streamlit": True, "pandas": True, "numpy": True, "xgboost": True,
        "scikit-learn": True, "nba_api": True, "scipy": True, "matplotlib": True,
        "shap": SHAP_AVAILABLE, "optuna": OPTUNA_AVAILABLE,
    }
    for pkg, ok in pkgs.items():
        st.markdown(f"- `{pkg}` {'✅' if ok else '❌ — `pip install ' + pkg + '`'}")
    st.divider()

    st.markdown("#### 🗑️ Data Management")
    d1, d2, d3, d4 = st.columns(4)
    if d1.button("🗑️ Clear Model",   use_container_width=True):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
            st.success("Cleared model.")
    if d2.button("🗑️ Clear Dataset", use_container_width=True):
        if os.path.exists(DATASET_FILE):
            os.remove(DATASET_FILE)
            st.success("Cleared dataset.")
    if d3.button("🗑️ Reset Elo",     use_container_width=True):
        if os.path.exists(ELO_FILE):
            os.remove(ELO_FILE)
            st.success("Elo reset.")
    if d4.button("🗑️ Clear Cache",   use_container_width=True):
        cache.clear_all()
        st.success("API cache cleared.")
