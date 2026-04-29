import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (DATASET_FILE, FEATURE_COLS, MODEL_FILE, OPTUNA_PARAMS_FILE)
from data.injuries import apply_injury_adjustment, calc_strength_factor
from engine.simulate import run_monte_carlo


# ── dataset ───────────────────────────────────────────────────────────────────

def build_master_dataset(seasons=None, status_cb=None):
    def report(msg):
        print(msg) if status_cb is None else status_cb(msg)

    if seasons is None:
        seasons = ["2024-25", "2025-26"]

    all_games = []
    for s in seasons:
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=s, league_id_nullable="00"
            )
            all_games.append(finder.get_data_frames()[0])
            time.sleep(1.0)
        except Exception:
            pass
    if not all_games:
        return None

    df = pd.concat(all_games, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"]   = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
    df["RESULT"]    = df["WL"].apply(lambda x: 1 if x == "W" else 0)
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])

    df["eFG_PCT"]  = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, 1)
    df["AST_TOV"]  = df["AST"] / df["TOV"].replace(0, 1)
    df["TS_PCT"]   = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"].replace(0, 1))).replace(0, 1)
    df["OREB_PCT"] = df["OREB"] / (df["OREB"] + df["DREB"]).replace(0, 1)

    def calc_streak(s):
        out, v = np.zeros(len(s)), 0
        for i, x in enumerate(s):
            v = (v + 1 if v >= 0 else 1) if x == 1 else (v - 1 if v <= 0 else -1)
            out[i] = v
        return pd.Series(out, index=s.index)

    df["WIN_STREAK"] = df.groupby("TEAM_ID")["RESULT"].transform(calc_streak)
    df["WIN_STREAK"] = df.groupby("TEAM_ID")["WIN_STREAK"].shift(1)
    df["PREV_DATE"]  = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["DAYS_REST"]  = (df["GAME_DATE"] - df["PREV_DATE"]).dt.days.fillna(3).clip(1, 10)
    df["IS_B2B"]     = (df["DAYS_REST"] <= 1).astype(int)

    for col in ["PLUS_MINUS", "PTS", "REB", "eFG_PCT", "AST_TOV", "TS_PCT", "OREB_PCT"]:
        df[col] = df.groupby("TEAM_ID")[col].transform(
            lambda x: x.shift(1).rolling(7, min_periods=3).mean()
        )

    df = df.dropna(subset=FEATURE_COLS + ["RESULT"])
    final = df[["GAME_DATE", "TEAM_ID"] + FEATURE_COLS + ["RESULT"]]
    final.to_csv(DATASET_FILE, index=False)
    report(f"Dataset built: {len(final):,} rows saved to {DATASET_FILE}")
    return final


# ── training ──────────────────────────────────────────────────────────────────

def train_model(force_rebuild: bool = False, use_optuna: bool = False,
                status_cb=None) -> tuple:
    """
    Train or reload the XGBoost model.
    Returns (model, row_count, metrics) or (None, 0, None) on failure.
    status_cb: optional callable(str) for progress messages.
    """
    def report(msg):
        print(msg) if status_cb is None else status_cb(msg)

    needs_rebuild = force_rebuild or not os.path.exists(DATASET_FILE)
    df = None

    if not needs_rebuild:
        df = pd.read_csv(DATASET_FILE)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            report(f"Dataset missing columns {missing} — rebuilding")
            needs_rebuild = True
            df = None

    if needs_rebuild:
        report("Downloading 2 seasons of game data…")
        df = build_master_dataset(status_cb=status_cb)

    if df is None or df.empty:
        return None, 0, None

    X = df[FEATURE_COLS]
    y = df["RESULT"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

    if use_optuna:
        _tuner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optuna_tuner.py")
        report("Running Optuna HPO (30 trials)…")
        proc = subprocess.run(
            [sys.executable, _tuner, "30"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode == 0 and os.path.exists(OPTUNA_PARAMS_FILE):
            with open(OPTUNA_PARAMS_FILE) as f:
                bp = json.load(f)
            model = XGBClassifier(**bp, eval_metric="logloss",
                                  early_stopping_rounds=40, random_state=42, n_jobs=-1)
        else:
            report("Optuna failed — using default hyperparameters")
            model = _default_model()
    else:
        model = _default_model()

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    vp     = model.predict(X_val)
    vproba = model.predict_proba(X_val)[:, 1]
    metrics = {
        "val_acc":   accuracy_score(y_val, vp),
        "val_loss":  log_loss(y_val, vproba),
        "n_train":   len(X_tr),
        "n_val":     len(X_val),
        "best_iter": getattr(model, "best_iteration", model.n_estimators),
        "feat_imp":  dict(zip(FEATURE_COLS, model.feature_importances_)),
        "val_proba": vproba.tolist(),
        "val_y":     y_val.tolist(),
    }
    dump(model, MODEL_FILE)
    return model, len(df), metrics


def _default_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=5,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        eval_metric="logloss", early_stopping_rounds=40,
        random_state=42, n_jobs=-1,
    )


# ── prediction ────────────────────────────────────────────────────────────────

def predict_hybrid(hd: dict, ad: dict, model,
                   h_inj=None, a_inj=None) -> tuple:
    hf  = calc_strength_factor(h_inj) if h_inj else 1.0
    af  = calc_strength_factor(a_inj) if a_inj else 1.0
    hs  = apply_injury_adjustment(hd, hf)
    as_ = apply_injury_adjustment(ad, af)

    def _row(s, is_home):
        return pd.DataFrame([[
            is_home, s["PLUS_MINUS"], s["PTS"], s["REB"],
            s["eFG_PCT"], s["AST_TOV"], s["TS_PCT"],
            s["WIN_STREAK"], s["OREB_PCT"], s["DAYS_REST"], s["IS_B2B"],
        ]], columns=FEATURE_COLS)

    xgb_h = model.predict_proba(_row(hs,  1))[0][1]
    xgb_a = model.predict_proba(_row(as_, 0))[0][1]
    xgb_p = xgb_h / (xgb_h + xgb_a)

    mc    = run_monte_carlo(hs, as_)
    mc_p  = mc["mc_home_prob"]
    fp    = xgb_p * 0.6 + mc_p * 0.4
    conflict = (xgb_p > 0.5) != (mc_p > 0.5)
    return fp, mc["spread_margin"], mc["median_total"], xgb_p, mc_p, conflict, mc, hf, af
