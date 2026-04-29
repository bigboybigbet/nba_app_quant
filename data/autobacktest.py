import json
import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams_static

from config import DATASET_FILE, FEATURE_COLS
from engine.features import get_team_advanced_stats
from engine.models import predict_hybrid, train_model

AUTOBACKTEST_LOG = "autobacktest_log.json"
SEASONS = ["2024-25", "2025-26"]

_TEAM_MAP = {t["id"]: t["full_name"] for t in nba_teams_static.get_teams()}


def _team_name(team_id: int) -> str | None:
    return _TEAM_MAP.get(int(team_id))


def load_log() -> set:
    if os.path.exists(AUTOBACKTEST_LOG):
        with open(AUTOBACKTEST_LOG) as f:
            return set(json.load(f).get("processed_game_ids", []))
    return set()


def _save_log(new_ids: set):
    existing = load_log()
    all_ids = existing | new_ids
    with open(AUTOBACKTEST_LOG, "w") as f:
        json.dump({"processed_game_ids": list(all_ids), "total": len(all_ids)}, f)


def log_stats() -> dict:
    if os.path.exists(AUTOBACKTEST_LOG):
        with open(AUTOBACKTEST_LOG) as f:
            d = json.load(f)
        return {"total": d.get("total", 0)}
    return {"total": 0}


def _fetch_new_games(status_cb=None) -> pd.DataFrame:
    def report(msg):
        if status_cb:
            status_cb(msg)

    processed = load_log()
    all_games = []

    for s in SEASONS:
        try:
            report(f"Fetching {s} schedule from NBA API…")
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=s, league_id_nullable="00"
            )
            all_games.append(finder.get_data_frames()[0])
            time.sleep(1.0)
        except Exception as e:
            report(f"Could not fetch {s}: {e}")

    if not all_games:
        return pd.DataFrame()

    df = pd.concat(all_games, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[df["WL"].isin(["W", "L"])]

    home_df = df[df["MATCHUP"].str.contains(r"vs\.", na=False)].copy()
    away_df = df[df["MATCHUP"].str.contains("@", na=False)].copy()

    paired = home_df.merge(
        away_df[["GAME_ID", "TEAM_ID", "WL"]],
        on="GAME_ID", suffixes=("_h", "_a"),
    ).rename(columns={
        "TEAM_ID_h": "HOME_TEAM_ID", "TEAM_ID_a": "AWAY_TEAM_ID",
        "WL_h": "WL_HOME", "GAME_DATE": "GAME_DATE",
    })

    paired["HOME_WON"] = (paired["WL_HOME"] == "W").astype(int)
    paired = paired[~paired["GAME_ID"].isin(processed)]
    paired = paired.sort_values("GAME_DATE").reset_index(drop=True)

    report(f"Found {len(paired)} new completed games not yet backtested.")
    return paired[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_WON"]]


def run_auto_backtest(model, status_cb=None) -> dict:
    """
    Blindfold-predict all new completed games using only pre-game stats,
    feed wrong predictions back as training lessons, retrain the model.
    Returns summary dict.
    """
    def report(msg):
        if status_cb:
            status_cb(msg)

    new_games = _fetch_new_games(status_cb=status_cb)

    if new_games.empty:
        return {"new": 0, "correct": 0, "lessons": 0, "acc": 0.0, "metrics": None}

    total = len(new_games)
    correct = 0
    lessons = []
    processed_ids = set()

    for i, row in new_games.iterrows():
        game_date = row["GAME_DATE"].date()
        home_name = _team_name(row["HOME_TEAM_ID"])
        away_name = _team_name(row["AWAY_TEAM_ID"])

        if not home_name or not away_name:
            continue

        hd = get_team_advanced_stats(home_name, target_date=game_date)
        ad = get_team_advanced_stats(away_name, target_date=game_date)

        if not hd or not ad:
            continue

        try:
            fp, *_ = predict_hybrid(hd, ad, model)
            pred   = 1 if fp > 0.5 else 0
            actual = int(row["HOME_WON"])

            if pred == actual:
                correct += 1
            else:
                for is_home, stats, result in [(1, hd, actual), (0, ad, 1 - actual)]:
                    lesson = {
                        "GAME_DATE": str(game_date), "TEAM_ID": 0,
                        "IS_HOME": is_home,
                        "PLUS_MINUS": stats["PLUS_MINUS"],
                        "PTS": stats["PTS"], "REB": stats["REB"],
                        "eFG_PCT": stats["eFG_PCT"], "AST_TOV": stats["AST_TOV"],
                        "TS_PCT": stats["TS_PCT"], "WIN_STREAK": stats["WIN_STREAK"],
                        "OREB_PCT": stats["OREB_PCT"], "DAYS_REST": stats["DAYS_REST"],
                        "IS_B2B": stats["IS_B2B"], "RESULT": result,
                    }
                    lessons.extend([lesson] * 3)

            processed_ids.add(row["GAME_ID"])
        except Exception:
            continue

        if (i + 1) % 25 == 0:
            report(f"  {i + 1}/{total} games processed…")

    actual_processed = len(processed_ids)
    metrics = None

    if lessons and os.path.exists(DATASET_FILE):
        wrong_count = len(lessons) // 6
        report(f"Injecting {wrong_count} wrong-prediction lessons into dataset…")
        master  = pd.read_csv(DATASET_FILE)
        updated = pd.concat([master, pd.DataFrame(lessons)], ignore_index=True)
        updated.to_csv(DATASET_FILE, index=False)

        report("Retraining model on updated dataset…")
        _, _, metrics = train_model(force_rebuild=False, status_cb=status_cb)

    _save_log(processed_ids)

    acc = correct / actual_processed if actual_processed > 0 else 0.0
    report(
        f"Done — {actual_processed} new games · "
        f"{correct}/{actual_processed} correct ({acc * 100:.1f}%)"
    )

    return {
        "new":     actual_processed,
        "correct": correct,
        "lessons": len(lessons) // 6,
        "acc":     acc,
        "metrics": metrics,
    }
