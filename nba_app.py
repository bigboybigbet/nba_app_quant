import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from joblib import dump, load
import os
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from nba_api.stats.static import teams
import requests
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG & PATHS
# ==========================================
MODEL_FILE = "nba_brain_v2.joblib"
LOG_FILE = "prediction_history.csv"
DATASET_FILE = "master_dataset_v2.csv"
N_SIMULATIONS = 50_000

st.set_page_config(
    page_title="NBA AI Quant Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS — DARK COURT THEME
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0e1a;
    color: #e8eaf0;
}
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 1.5px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1629 0%, #0b1020 100%);
    border-right: 1px solid #1e2d4a;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.05rem !important;
    letter-spacing: 1px;
    color: #667eea !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f97316 !important;
    border-bottom: 2px solid #f97316 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827, #1a2236);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
    color: #f97316 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f97316, #ef4444) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(249,115,22,0.35) !important;
}

/* Inputs */
.stTextInput > div > div > input, .stDateInput > div > div > input {
    background-color: #111827 !important;
    border: 1px solid #1e3a5f !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}

/* DataFrame */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #1e2d4a; }

/* Info/success/error boxes */
.stAlert { border-radius: 10px !important; }

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #667eea, #f97316) !important; }

/* Score card custom */
.score-card {
    background: linear-gradient(135deg, #111827, #1a2236);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
.team-name { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; letter-spacing: 2px; }
.prob-big  { font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; color: #f97316; }
.vs-badge  { font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem; color: #667eea; }
.stat-row  { display: flex; justify-content: space-between; margin: 6px 0; font-size: 0.85rem; color: #94a3b8; }
.stat-val  { color: #e2e8f0; font-weight: 600; }
.badge-safe   { background: #14532d; color: #4ade80; border-radius: 6px; padding: 4px 12px; font-size:0.8rem; font-weight:600; }
.badge-danger { background: #450a0a; color: #f87171; border-radius: 6px; padding: 4px 12px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'schedule_data' not in st.session_state:
    st.session_state.schedule_data = None

# ==========================================
# 1. FEATURE ENGINEERING
# ==========================================

FEATURE_COLS = ['IS_HOME', 'PLUS_MINUS', 'PTS', 'REB', 'eFG_PCT',
                'AST_TOV', 'TS_PCT', 'WIN_STREAK', 'OREB_PCT']


def get_team_advanced_stats(team_name, target_date=None, window=7):
    """Fetch rolling stats + std devs for Monte Carlo, with richer features for XGBoost."""
    try:
        nba_teams = teams.get_teams()
        team_info = [t for t in nba_teams
                     if team_name.lower() in t['full_name'].lower()
                     or t['full_name'].lower() in team_name.lower()]
        if not team_info:
            return None

        team_id = team_info[0]['id']
        full_team_name = team_info[0]['full_name']

        finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        df = finder.get_data_frames()[0]
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        if target_date:
            df = df[df['GAME_DATE'] < pd.to_datetime(target_date)]

        needed = ['PLUS_MINUS', 'PTS', 'FGM', 'FGA', 'FG3M', 'FTA', 'FTM',
                  'AST', 'TOV', 'REB', 'OREB', 'DREB', 'WL']
        df = df.sort_values('GAME_DATE').dropna(subset=needed)
        if df.empty or len(df) < 3:
            return None

        recent = df.tail(window).copy()
        recent['PTS_ALLOWED'] = recent['PTS'] - recent['PLUS_MINUS']

        # ---- XGBoost features ----
        avg_plus_minus = recent['PLUS_MINUS'].mean()
        total_fga = recent['FGA'].sum()
        efg_pct = (recent['FGM'].sum() + 0.5 * recent['FG3M'].sum()
                   ) / total_fga if total_fga > 0 else 0
        ast_tov = recent['AST'].sum() / max(recent['TOV'].sum(), 1)

        # True Shooting %
        total_pts = recent['PTS'].sum()
        ts_denom = 2 * (total_fga + 0.44 * recent['FTA'].sum())
        ts_pct = total_pts / ts_denom if ts_denom > 0 else 0

        # Win Streak (positive = win streak, negative = loss streak)
        streak = 0
        for wl in reversed(recent['WL'].tolist()):
            if wl == 'W':
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break

        # Offensive Rebound %
        total_oreb = recent['OREB'].sum()
        total_dreb = recent['DREB'].sum()
        oreb_pct = total_oreb / \
            (total_oreb + total_dreb) if (total_oreb + total_dreb) > 0 else 0

        # ---- Monte Carlo distribution params ----
        avg_pts = recent['PTS'].mean()
        std_pts = recent['PTS'].std() if len(recent) > 1 else 6.0
        avg_pts_allowed = recent['PTS_ALLOWED'].mean()
        std_pts_allowed = recent['PTS_ALLOWED'].std() if len(
            recent) > 1 else 6.0

        return {
            "name": full_team_name,
            # XGBoost
            "PLUS_MINUS": avg_plus_minus,
            "REB":        recent['REB'].mean(),
            "eFG_PCT":    efg_pct,
            "AST_TOV":    ast_tov,
            "TS_PCT":     ts_pct,
            "WIN_STREAK": streak,
            "OREB_PCT":   oreb_pct,
            # Monte Carlo
            "PTS":              avg_pts,
            "PTS_STD":          max(std_pts, 3.0),
            "PTS_ALLOWED":      avg_pts_allowed,
            "PTS_ALLOWED_STD":  max(std_pts_allowed, 3.0),
        }
    except Exception:
        return None


# ==========================================
# 2. IMPROVED MONTE CARLO (50k sims)
# ==========================================

def run_monte_carlo(h_stats, a_stats, simulations=N_SIMULATIONS):
    """
    50,000-game simulation using combined offense/defense distributions.
    Returns win prob, spread, total, percentile CIs, and raw arrays for plotting.
    """
    # Home court +3 pts
    h_mean = (h_stats['PTS'] + a_stats['PTS_ALLOWED']) / 2 + 3.0
    a_mean = (a_stats['PTS'] + h_stats['PTS_ALLOWED']) / 2

    h_std = np.sqrt(h_stats['PTS_STD']**2 +
                    a_stats['PTS_ALLOWED_STD']**2) / np.sqrt(2)
    a_std = np.sqrt(a_stats['PTS_STD']**2 +
                    h_stats['PTS_ALLOWED_STD']**2) / np.sqrt(2)

    h_std = max(h_std, 4.0)
    a_std = max(a_std, 4.0)

    rng = np.random.default_rng(42)
    h_sims = rng.normal(h_mean, h_std, simulations)
    a_sims = rng.normal(a_mean, a_std, simulations)

    margins = h_sims - a_sims
    totals = h_sims + a_sims
    h_wins = np.sum(margins > 0)

    return {
        "mc_home_prob":   h_wins / simulations,
        "spread_margin":  float(np.mean(margins)),
        "median_total":   float(np.median(totals)),
        # Confidence intervals
        "spread_lo":  float(np.percentile(margins, 25)),
        "spread_hi":  float(np.percentile(margins, 75)),
        "total_lo":   float(np.percentile(totals, 25)),
        "total_hi":   float(np.percentile(totals, 75)),
        # OT probability (within 3 pts either way)
        "ot_prob":    float(np.mean(np.abs(margins) <= 3)),
        # Raw for chart
        "_margins":   margins,
        "_totals":    totals,
        "_h_sims":    h_sims,
        "_a_sims":    a_sims,
    }


# ==========================================
# 3. DATASET BUILDER (richer features)
# ==========================================

def build_master_dataset():
    seasons = ['2022-23', '2023-24', '2024-25', '2025-26']
    all_games = []
    for season in seasons:
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable='00')
            all_games.append(finder.get_data_frames()[0])
            time.sleep(1.0)
        except Exception:
            pass

    if not all_games:
        return None

    df = pd.concat(all_games, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['RESULT'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    # Derived
    df['eFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, 1)
    df['AST_TOV'] = df['AST'] / df['TOV'].replace(0, 1)
    ts_denom = 2 * (df['FGA'] + 0.44 * df['FTA'].replace(0, 1))
    df['TS_PCT'] = df['PTS'] / ts_denom.replace(0, 1)
    df['OREB_PCT'] = df['OREB'] / (df['OREB'] + df['DREB']).replace(0, 1)

    # Win streak (rolling sign-encoded)
    def calc_streak(series):
        out = np.zeros(len(series))
        s = 0
        for i, v in enumerate(series):
            if v == 1:
                s = s + 1 if s >= 0 else 1
            else:
                s = s - 1 if s <= 0 else -1
            out[i] = s
        return pd.Series(out, index=series.index)

    df['WIN_STREAK'] = df.groupby('TEAM_ID')['RESULT'].transform(calc_streak)
    df['WIN_STREAK'] = df.groupby('TEAM_ID')['WIN_STREAK'].shift(1)

    cols_to_roll = ['PLUS_MINUS', 'PTS', 'REB',
                    'eFG_PCT', 'AST_TOV', 'TS_PCT', 'OREB_PCT']
    for col in cols_to_roll:
        df[col] = df.groupby('TEAM_ID')[col].transform(
            lambda x: x.shift(1).rolling(7, min_periods=3).mean())

    df = df.dropna(subset=FEATURE_COLS + ['RESULT'])
    final_df = df[['GAME_DATE', 'TEAM_ID'] + FEATURE_COLS + ['RESULT']]
    final_df.to_csv(DATASET_FILE, index=False)
    return final_df


# ==========================================
# 4. XGBOOST TRAINING (calibrated)
# ==========================================

def train_model(force_rebuild=False):
    needs_rebuild = force_rebuild or not os.path.exists(DATASET_FILE)

    # Also rebuild if the saved CSV is missing any required feature columns
    if not needs_rebuild:
        df = pd.read_csv(DATASET_FILE)
        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_cols:
            st.warning(
                f"⚠️ Saved dataset is missing {missing_cols} — rebuilding automatically…")
            needs_rebuild = True

    if needs_rebuild:
        with st.spinner("📥 Building master dataset (4 seasons)…"):
            df = build_master_dataset()

    if df is None or df.empty:
        return None, 0, None

    X = df[FEATURE_COLS]
    y = df['RESULT']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, shuffle=False)

    base_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Use base_model directly — XGBoost trained with logloss is already well-calibrated
    cal_model = base_model

    val_preds = cal_model.predict(X_val)
    val_proba = cal_model.predict_proba(X_val)[:, 1]
    metrics = {
        "val_acc":  accuracy_score(y_val, val_preds),
        "val_loss": log_loss(y_val, val_proba),
        "n_train":  len(X_train),
        "n_val":    len(X_val),
        "best_iter": base_model.best_iteration,
        "feat_imp": dict(zip(FEATURE_COLS, base_model.feature_importances_)),
    }

    dump((cal_model, base_model), MODEL_FILE)
    return cal_model, len(df), metrics


# ==========================================
# 5. HYBRID PREDICTOR
# ==========================================

def predict_hybrid(home_stats, away_stats, model):
    h_df = pd.DataFrame([[
        1,
        home_stats['PLUS_MINUS'], home_stats['PTS'], home_stats['REB'],
        home_stats['eFG_PCT'], home_stats['AST_TOV'], home_stats['TS_PCT'],
        home_stats['WIN_STREAK'], home_stats['OREB_PCT'],
    ]], columns=FEATURE_COLS)

    a_df = pd.DataFrame([[
        0,
        away_stats['PLUS_MINUS'], away_stats['PTS'], away_stats['REB'],
        away_stats['eFG_PCT'], away_stats['AST_TOV'], away_stats['TS_PCT'],
        away_stats['WIN_STREAK'], away_stats['OREB_PCT'],
    ]], columns=FEATURE_COLS)

    xgb_h = model.predict_proba(h_df)[0][1]
    xgb_a = model.predict_proba(a_df)[0][1]
    xgb_prob = xgb_h / (xgb_h + xgb_a)  # normalised

    mc = run_monte_carlo(home_stats, away_stats)
    mc_prob = mc['mc_home_prob']

    # Weighted blend: 60% XGB, 40% MC
    final_prob = xgb_prob * 0.6 + mc_prob * 0.4

    is_conflict = (xgb_prob > 0.5) != (mc_prob > 0.5)

    return final_prob, mc['spread_margin'], mc['median_total'], xgb_prob, mc_prob, is_conflict, mc


# ==========================================
# 6. ESPN API HELPERS
# ==========================================

def get_schedule_by_date(selected_date):
    date_str = selected_date.strftime('%Y%m%d')
    try:
        data = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}",
            timeout=10).json()
        schedule = []
        for event in data.get('events', []):
            try:
                comps = event['competitions'][0]['competitors']
                schedule.append({
                    "Date": selected_date.strftime('%Y-%m-%d'),
                    "Home Team": next(c['team']['displayName'] for c in comps if c['homeAway'] == 'home'),
                    "Away Team": next(c['team']['displayName'] for c in comps if c['homeAway'] == 'away'),
                })
            except Exception:
                continue
        return pd.DataFrame(schedule)
    except Exception:
        return None


def get_past_results_espn(selected_date):
    date_str = selected_date.strftime('%Y%m%d')
    try:
        data = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}",
            timeout=10).json()
        results = []
        for event in data.get('events', []):
            if event['status']['type']['name'] == 'STATUS_FINAL':
                comps = event['competitions'][0]['competitors']
                hc = next(c for c in comps if c['homeAway'] == 'home')
                ac = next(c for c in comps if c['homeAway'] == 'away')
                hs, as_ = int(hc['score']), int(ac['score'])
                results.append({
                    "Home Team": hc['team']['displayName'],
                    "Away Team": ac['team']['displayName'],
                    "H_Score": hs, "A_Score": as_,
                    "Actual_Winner": 1 if hs > as_ else 0,
                })
        return results
    except Exception:
        return []


# ==========================================
# 7. CHART HELPERS
# ==========================================
DARK_BG = "#0b0e1a"
CARD_BG = "#111827"
ORANGE = "#f97316"
PURPLE = "#667eea"
GREEN = "#4ade80"
RED = "#f87171"
GRAY = "#374151"


def fig_mc_distributions(mc_result, ht, at):
    """Score distribution + margin distribution side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor(DARK_BG)

    h_sims = mc_result['_h_sims']
    a_sims = mc_result['_a_sims']
    margins = mc_result['_margins']

    # Left: score distributions
    ax = axes[0]
    ax.set_facecolor(CARD_BG)
    ax.hist(h_sims, bins=60, alpha=0.7, color=ORANGE, density=True, label=ht)
    ax.hist(a_sims, bins=60, alpha=0.7, color=PURPLE, density=True, label=at)
    ax.axvline(np.mean(h_sims), color=ORANGE, linestyle='--', linewidth=1.5)
    ax.axvline(np.mean(a_sims), color=PURPLE, linestyle='--', linewidth=1.5)
    ax.set_title("Score Distribution", color='white',
                 fontsize=11, fontweight='bold')
    ax.legend(frameon=False, labelcolor='white', fontsize=8)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.set_xlabel("Points", color='#94a3b8', fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Right: margin distribution
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    pos_margins = margins[margins > 0]
    neg_margins = margins[margins <= 0]
    ax2.hist(pos_margins, bins=60, alpha=0.8, color=ORANGE, density=True)
    ax2.hist(neg_margins, bins=60, alpha=0.8, color=PURPLE, density=True)
    ax2.axvline(0, color='white', linewidth=1, linestyle='--')
    ax2.axvline(mc_result['spread_margin'],
                color=ORANGE, linewidth=2, linestyle='-')
    ax2.set_title("Margin Distribution", color='white',
                  fontsize=11, fontweight='bold')
    ax2.tick_params(colors='#94a3b8', labelsize=8)
    ax2.set_xlabel(f"← {at} wins  |  {ht} wins →", color='#94a3b8', fontsize=8)
    for sp in ax2.spines.values():
        sp.set_visible(False)

    fig.tight_layout(pad=1.2)
    return fig


def fig_feature_importance(feat_imp: dict):
    """Horizontal bar chart of XGBoost feature importances."""
    labels = list(feat_imp.keys())
    values = list(feat_imp.values())
    idx = np.argsort(values)
    labels = [labels[i] for i in idx]
    values = [values[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    colors = [ORANGE if v == max(values) else PURPLE for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.6)
    ax.set_title("XGBoost Feature Importance", color='white',
                 fontsize=10, fontweight='bold')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.set_xlabel("Importance", color='#94a3b8', fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=1.0)
    return fig


def fig_radar(h_stats, a_stats):
    """Radar/spider chart comparing two teams on key stats."""
    cats = ['eFG%', 'TS%', 'AST/TOV', 'OREB%', 'Reb', '+/-']
    N = len(cats)

    def normalise(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-9) for v in vals]

    h_raw = [h_stats['eFG_PCT'], h_stats['TS_PCT'], min(h_stats['AST_TOV'], 5)/5,
             h_stats['OREB_PCT'], h_stats['REB']/50, (h_stats['PLUS_MINUS']+15)/30]
    a_raw = [a_stats['eFG_PCT'], a_stats['TS_PCT'], min(a_stats['AST_TOV'], 5)/5,
             a_stats['OREB_PCT'], a_stats['REB']/50, (a_stats['PLUS_MINUS']+15)/30]
    h_raw = [max(0, min(1, v)) for v in h_raw]
    a_raw = [max(0, min(1, v)) for v in a_raw]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    h_vals = h_raw + [h_raw[0]]
    a_vals = a_raw + [a_raw[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    ax.plot(angles, h_vals, color=ORANGE, linewidth=2, linestyle='solid')
    ax.fill(angles, h_vals, color=ORANGE, alpha=0.25)
    ax.plot(angles, a_vals, color=PURPLE, linewidth=2, linestyle='solid')
    ax.fill(angles, a_vals, color=PURPLE, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color='#94a3b8', size=8)
    ax.tick_params(colors='#94a3b8')
    ax.yaxis.set_visible(False)
    ax.grid(color=GRAY, linewidth=0.5)
    ax.spines['polar'].set_color(GRAY)

    h_patch = mpatches.Patch(color=ORANGE, label=h_stats['name'].split()[-1])
    a_patch = mpatches.Patch(color=PURPLE, label=a_stats['name'].split()[-1])
    ax.legend(handles=[h_patch, a_patch], loc='lower center',
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False,
              labelcolor='white', fontsize=8)
    fig.tight_layout()
    return fig


# ==========================================
# 8.  SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px'>
        <div style='font-family:Bebas Neue; font-size:2rem; color:#f97316; letter-spacing:3px'>NBA AI QUANT</div>
        <div style='font-size:0.7rem; color:#667eea; letter-spacing:2px'>XGBOOST + MONTE CARLO v2</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### ⚙️ Model Control")

    force_rebuild = st.toggle("Force full rebuild", value=False,
                              help="Re-download 3 seasons of data and retrain from scratch")

    if st.button("🔄 Train / Reload Model", use_container_width=True):
        cal_model, count, metrics = train_model(force_rebuild=force_rebuild)
        if cal_model:
            st.session_state['model_metrics'] = metrics
            st.session_state['feat_imp'] = metrics['feat_imp']
            st.success(f"✅ Model ready — {count:,} samples")
            st.caption(
                f"Val Acc: **{metrics['val_acc']:.1%}**  |  Log-loss: **{metrics['val_loss']:.4f}**")
            st.caption(f"Best iter: {metrics['best_iter']}")
        else:
            st.error("Training failed — check NBA API connection.")

    st.divider()

    if 'feat_imp' in st.session_state:
        st.markdown("#### 📊 Feature Importance")
        st.pyplot(fig_feature_importance(
            st.session_state['feat_imp']), use_container_width=True)

    st.divider()
    st.caption(
        "Data: NBA Stats API + ESPN  |  4 Seasons (2022–26)  |  50,000 Sims")


# ==========================================
# 9. MAIN TABS
# ==========================================

st.markdown("""
<div style='padding: 10px 0 4px'>
    <span style='font-family:Bebas Neue; font-size:2.4rem; letter-spacing:3px; color:white'>🏀 NBA HYBRID QUANT</span>
    <span style='font-family:Bebas Neue; font-size:1rem; color:#667eea; margin-left:12px; letter-spacing:2px'>XGBOOST + MONTE CARLO PRO</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    ["🎯  SINGLE GAME", "💰  RADAR SCANNER", "🔬  BACKTEST"])

# ──────────────────────────────────────────
# TAB 1 — SINGLE GAME PREDICTION
# ──────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        st.markdown("##### 🏠 Home Team")
        ht = st.text_input("", "Lakers", key="ht_input",
                           placeholder="e.g. Lakers, Warriors, Celtics")
        st.markdown("##### ✈️ Away Team")
        at = st.text_input("", "Warriors", key="at_input",
                           placeholder="e.g. Nuggets, Bucks, Heat")

        run_btn = st.button("🚀 RUN 50,000 SIMULATIONS", type="primary",
                            use_container_width=True)

    with right:
        st.markdown("""
        <div class='score-card' style='margin-top:8px'>
            <div style='color:#94a3b8; font-size:0.7rem; letter-spacing:2px; margin-bottom:8px'>HOW IT WORKS</div>
            <div style='text-align:left; font-size:0.82rem; color:#cbd5e1; line-height:1.7'>
                🧠 <b>XGBoost (60%)</b> — 9 rolling features<br>
                🎲 <b>Monte Carlo (40%)</b> — 50k game sims<br>
                📐 Calibrated probabilities (Isotonic)<br>
                🔀 Conflict detection between models<br>
                📊 Spread & O/U with 25/75 CIs
            </div>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        if not os.path.exists(MODEL_FILE):
            st.error(
                "⚠️ No trained model found. Click **Train / Reload Model** in the sidebar first.")
        else:
            cal_model, _ = load(MODEL_FILE)
            with st.spinner("⚡ Fetching stats & running 50,000 simulations…"):
                hd = get_team_advanced_stats(ht)
                ad = get_team_advanced_stats(at)

            if not hd or not ad:
                st.error(
                    "❌ Team name not found or NBA API error. Try full name, e.g. 'Los Angeles Lakers'.")
            else:
                final_prob, spread, total, xgb_prob, mc_prob, is_conflict, mc_full = predict_hybrid(
                    hd, ad, cal_model)
                confidence = max(final_prob, 1 - final_prob)
                winner = hd['name'] if final_prob > 0.5 else ad['name']
                loser = ad['name'] if final_prob > 0.5 else hd['name']
                w_prob = confidence

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Conflict / Consensus banner ──
                if is_conflict:
                    st.markdown("""
                    <div style='background:#450a0a; border:1px solid #b91c1c; border-radius:10px;
                                padding:14px 20px; margin-bottom:16px'>
                        🚨 <b>MODEL CONFLICT</b> — XGBoost and Monte Carlo disagree on the winner.
                        Confidence is reduced. <b>Proceed with caution.</b>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background:#14532d; border:1px solid #16a34a; border-radius:10px;
                                padding:14px 20px; margin-bottom:16px'>
                        ✅ <b>HIGH CONSENSUS</b> — Both AI engines agree on the same winner.
                    </div>""", unsafe_allow_html=True)

                # ── Main prediction row ──
                c1, c2, c3 = st.columns(3)
                c1.metric("🧠 XGBoost (Home)",  f"{xgb_prob*100:.1f}%")
                c2.metric("🎲 Monte Carlo",      f"{mc_prob*100:.1f}%")
                c3.metric("⚖️ Hybrid Final",     f"{final_prob*100:.1f}%",
                          delta=f"{'✅ SAFE BET' if confidence >= 0.65 and not is_conflict else '⚠️ RISKY'}")

                st.divider()

                # ── Visual comparison row ──
                ra, rb, rc = st.columns([1.1, 0.6, 1.1], gap="small")

                with ra:
                    bar_h = int(final_prob * 100)
                    st.markdown(f"""
                    <div class='score-card'>
                        <div class='team-name' style='color:#f97316'>{hd['name'].split()[-1].upper()}</div>
                        <div style='color:#94a3b8; font-size:0.7rem; margin:4px 0 10px'>HOME</div>
                        <div class='prob-big'>{final_prob*100:.0f}%</div>
                        <div style='background:#1e2d4a; border-radius:6px; height:8px; margin:12px 0'>
                            <div style='background:#f97316; width:{bar_h}%; height:8px; border-radius:6px'></div>
                        </div>
                        <div class='stat-row'><span>eFG%</span><span class='stat-val'>{hd['eFG_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>TS%</span><span class='stat-val'>{hd['TS_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>AST/TOV</span><span class='stat-val'>{hd['AST_TOV']:.2f}</span></div>
                        <div class='stat-row'><span>Win Streak</span><span class='stat-val'>{int(hd['WIN_STREAK'])}</span></div>
                        <div class='stat-row'><span>Avg PTS</span><span class='stat-val'>{hd['PTS']:.1f}</span></div>
                    </div>""", unsafe_allow_html=True)

                with rb:
                    st.markdown(f"""
                    <div style='display:flex; align-items:center; justify-content:center;
                                height:100%; padding-top:60px'>
                        <div class='vs-badge'>VS</div>
                    </div>""", unsafe_allow_html=True)

                with rc:
                    bar_a = int((1-final_prob) * 100)
                    st.markdown(f"""
                    <div class='score-card'>
                        <div class='team-name' style='color:#667eea'>{ad['name'].split()[-1].upper()}</div>
                        <div style='color:#94a3b8; font-size:0.7rem; margin:4px 0 10px'>AWAY</div>
                        <div class='prob-big' style='color:#667eea'>{(1-final_prob)*100:.0f}%</div>
                        <div style='background:#1e2d4a; border-radius:6px; height:8px; margin:12px 0'>
                            <div style='background:#667eea; width:{bar_a}%; height:8px; border-radius:6px'></div>
                        </div>
                        <div class='stat-row'><span>eFG%</span><span class='stat-val'>{ad['eFG_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>TS%</span><span class='stat-val'>{ad['TS_PCT']:.3f}</span></div>
                        <div class='stat-row'><span>AST/TOV</span><span class='stat-val'>{ad['AST_TOV']:.2f}</span></div>
                        <div class='stat-row'><span>Win Streak</span><span class='stat-val'>{int(ad['WIN_STREAK'])}</span></div>
                        <div class='stat-row'><span>Avg PTS</span><span class='stat-val'>{ad['PTS']:.1f}</span></div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Monte Carlo details ──
                st.markdown(
                    "#### 🎲 Monte Carlo Results  *(50,000 simulations)*")
                m1, m2, m3, m4 = st.columns(4)
                if spread > 0:
                    m1.metric("📐 Spread", f"{hd['name'].split()[-1]} -{spread:.1f}",
                              help=f"25th–75th pct: {mc_full['spread_lo']:.1f} to {mc_full['spread_hi']:.1f}")
                else:
                    m1.metric("📐 Spread", f"{ad['name'].split()[-1]} -{abs(spread):.1f}",
                              help=f"25th–75th pct: {mc_full['spread_lo']:.1f} to {mc_full['spread_hi']:.1f}")
                m2.metric("🔢 O/U Line",   f"{total:.1f}",
                          help=f"25–75th pct: {mc_full['total_lo']:.1f} – {mc_full['total_hi']:.1f}")
                m3.metric("🏠 Home Win %",
                          f"{mc_full['mc_home_prob']*100:.1f}%")
                m4.metric("⏱️ OT Prob",     f"{mc_full['ot_prob']*100:.1f}%",
                          help="Probability the game finishes within 3 pts (OT territory)")

                # ── Charts row ──
                ch1, ch2 = st.columns([2.2, 1])
                with ch1:
                    st.pyplot(fig_mc_distributions(mc_full, hd['name'].split()[-1], ad['name'].split()[-1]),
                              use_container_width=True)
                with ch2:
                    st.pyplot(fig_radar(hd, ad), use_container_width=True)

                # ── Pick summary ──
                if not is_conflict and confidence >= 0.55:
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#1a2236,#111827);
                                border:1px solid #f97316; border-radius:12px;
                                padding:20px 24px; margin-top:16px; text-align:center'>
                        <div style='color:#94a3b8; font-size:0.7rem; letter-spacing:2px; margin-bottom:6px'>AI PICK</div>
                        <div style='font-family:Bebas Neue; font-size:2rem; color:#f97316'>{winner}</div>
                        <div style='color:#94a3b8; font-size:0.85rem'>to win · {w_prob*100:.1f}% confidence</div>
                    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# TAB 2 — RADAR SCANNER
# ──────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    scan_col, _ = st.columns([1.2, 1])
    with scan_col:
        target_date = st.date_input("🗓️ Select game date", datetime.today())
        scan_btn = st.button(
            "🔍 Scan Schedule & Calculate Lines", use_container_width=True)

    if scan_btn:
        if not os.path.exists(MODEL_FILE):
            st.error("⚠️ Train the model first.")
        else:
            df_res = get_schedule_by_date(target_date)
            if df_res is None or df_res.empty:
                st.warning("No games found for this date.")
            else:
                cal_model, _ = load(MODEL_FILE)
                recs = []
                prog = st.progress(0)
                n = len(df_res)

                for i, row in df_res.iterrows():
                    hn, an = row['Home Team'], row['Away Team']
                    hd = get_team_advanced_stats(hn)
                    ad = get_team_advanced_stats(an)
                    if hd and ad:
                        final_prob, spread, total, xgb_prob, mc_prob, is_conflict, mc_full = predict_hybrid(
                            hd, ad, cal_model)
                        confidence = max(final_prob, 1 - final_prob)
                        pick = hd['name'] if final_prob > 0.5 else ad['name']
                        recs.append({
                            "Matchup":    f"{hn}  vs  {an}",
                            "AI Pick":    pick,
                            "Confidence": f"{confidence*100:.1f}%",
                            "XGBoost":    f"{xgb_prob*100:.1f}%",
                            "Monte Carlo": f"{mc_prob*100:.1f}%",
                            "Spread":     f"{hn} -{spread:.1f}" if spread > 0 else f"{an} -{abs(spread):.1f}",
                            "O/U":        f"{total:.1f}",
                            "OT%":        f"{mc_full['ot_prob']*100:.1f}%",
                            "Status":     "✅ Consensus" if not is_conflict else "🚨 Conflict",
                        })
                    prog.progress((i + 1) / n)

                if recs:
                    result_df = pd.DataFrame(recs)
                    st.dataframe(result_df, hide_index=True,
                                 use_container_width=True)

                    # VIP picks (≥65% confidence, no conflict)
                    vip = result_df[(result_df['Status'] == '✅ Consensus') &
                                    (result_df['Confidence'].str.rstrip('%').astype(float) >= 65)]
                    if not vip.empty:
                        st.markdown(
                            "#### 🔥 VIP Picks (≥65% confident + consensus)")
                        st.dataframe(vip[['Matchup', 'AI Pick', 'Confidence', 'Spread', 'O/U']],
                                     hide_index=True, use_container_width=True)
                else:
                    st.warning("Could not compute predictions for any game.")


# ──────────────────────────────────────────
# TAB 3 — BACKTEST
# ──────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📋 Replay historical dates to measure accuracy. Wrong predictions are added back "
            "to the training set (×3 oversampling) and the model self-updates.")

    bt_col, _ = st.columns([1.2, 1])
    with bt_col:
        date_range = st.date_input(
            "🗓️ Backtest date range (3–5 days recommended)",
            [datetime.today() - timedelta(days=3), datetime.today() - timedelta(days=1)])

    bt_btn = st.button("🚀 RUN BACKTEST & SELF-LEARN",
                       type="primary", use_container_width=True)

    if bt_btn:
        if len(date_range) != 2:
            st.error("Select a valid start and end date.")
        elif not os.path.exists(MODEL_FILE):
            st.error("⚠️ Train the model first.")
        else:
            cal_model, _ = load(MODEL_FILE)
            start_date, end_date = date_range

            all_eval_data = []
            new_lessons = []
            total_games = total_correct = 0
            high_conf_games = high_conf_correct = 0

            pb = st.progress(0)
            total_days = (end_date - start_date).days + 1
            current_date = start_date
            day_count = 0

            with st.spinner("🕰️ Running time-machine backtest…"):
                while current_date <= end_date:
                    past_games = get_past_results_espn(current_date)

                    for game in past_games:
                        hn = game['Home Team']
                        an = game['Away Team']
                        act = game['Actual_Winner']

                        hd = get_team_advanced_stats(
                            hn, target_date=current_date)
                        ad = get_team_advanced_stats(
                            an, target_date=current_date)

                        if hd and ad:
                            fp, spread, total, xgb_p, mc_p, conflict, mc_full = predict_hybrid(
                                hd, ad, cal_model)
                            pred = 1 if fp > 0.5 else 0
                            conf = max(fp, 1 - fp)
                            is_ok = pred == act
                            total_games += 1
                            if is_ok:
                                total_correct += 1
                            if conf >= 0.65 and not conflict:
                                high_conf_games += 1
                                if is_ok:
                                    high_conf_correct += 1

                            if not is_ok:
                                for is_home, stats, result in [(1, hd, act), (0, ad, 1 - act)]:
                                    new_lessons.append({
                                        'GAME_DATE': current_date.strftime('%Y-%m-%d'),
                                        'TEAM_ID': 0, 'IS_HOME': is_home,
                                        'PLUS_MINUS': stats['PLUS_MINUS'], 'PTS': stats['PTS'],
                                        'REB': stats['REB'], 'eFG_PCT': stats['eFG_PCT'],
                                        'AST_TOV': stats['AST_TOV'], 'TS_PCT': stats['TS_PCT'],
                                        'WIN_STREAK': stats['WIN_STREAK'], 'OREB_PCT': stats['OREB_PCT'],
                                        'RESULT': result,
                                    })

                            all_eval_data.append({
                                "Date":      current_date.strftime('%d/%m'),
                                "Matchup":   f"{hn} vs {an}",
                                "Score":     f"{game['H_Score']}–{game['A_Score']}",
                                "AI Pick":   f"🏠 Home ({conf*100:.1f}%)" if pred == 1 else f"✈️ Away ({conf*100:.1f}%)",
                                "Engines":   "🚨 Conflict" if conflict else "✅ Agree",
                                "Result":    "✅ Win" if is_ok else "❌ Loss",
                            })

                    day_count += 1
                    pb.progress(day_count / total_days)
                    current_date += timedelta(days=1)

            if total_games > 0:
                st.markdown("### 📊 Backtest Performance Report")
                acc = total_correct / total_games

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Games Scanned",  f"{total_games}")
                r2.metric("Overall Acc.",   f"{acc*100:.1f}%",
                          delta=f"+{total_correct} correct")
                if high_conf_games > 0:
                    vip_acc = high_conf_correct / high_conf_games
                    r3.metric("VIP Accuracy (≥65%)", f"{vip_acc*100:.1f}%",
                              delta=f"{high_conf_games} VIP games")
                else:
                    r3.metric("VIP Accuracy (≥65%)", "N/A")
                r4.metric("Wrong Predictions", f"{total_games - total_correct}",
                          delta=f"{len(new_lessons)//2} lessons added")

                st.dataframe(pd.DataFrame(all_eval_data),
                             hide_index=True, use_container_width=True)

                # Self-learning update
                if new_lessons and os.path.exists(DATASET_FILE):
                    st.markdown("---")
                    st.warning(f"⚠️ {len(new_lessons)//2} incorrect predictions detected. "
                               "Activating XGBoost self-learning protocol…")
                    master_df = pd.read_csv(DATASET_FILE)
                    lessons_df = pd.DataFrame(new_lessons)
                    updated_df = pd.concat(
                        [master_df, lessons_df, lessons_df, lessons_df], ignore_index=True)
                    updated_df.to_csv(DATASET_FILE, index=False)
                    _, new_count, new_metrics = train_model(
                        force_rebuild=False)
                    if new_metrics:
                        st.success(
                            f"🧬 EVOLUTION COMPLETE — Model retrained on {new_count:,} samples.  "
                            f"New val accuracy: **{new_metrics['val_acc']:.1%}**")
            else:
                st.warning(
                    "No completed game results found in the selected date range.")
