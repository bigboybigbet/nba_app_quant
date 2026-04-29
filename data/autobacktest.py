import json
import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams as nba_teams_static

import data.cache as cache
from config import DATASET_FILE
from data.elo import get_team_elo
from engine.models import predict_hybrid, train_model

AUTOBACKTEST_LOG = "autobacktest_log.json"
SEASONS = ["2024-25", "2025-26"]

_TEAM_MAP = {t["id"]: t["full_name"] for t in nba_teams_static.get_teams()}


# ── log helpers ───────────────────────────────────────────────────────────────

def load_log() -> set:
    if os.path.exists(AUTOBACKTEST_LOG):
        with open(AUTOBACKTEST_LOG) as f:
            return set(json.load(f).get("processed_game_ids", []))
    return set()


def _save_log(new_ids: set):
    existing = load_log()
    all_ids  = existing | new_ids
    with open(AUTOBACKTEST_LOG, "w") as f:
        json.dump({"processed_game_ids": list(all_ids), "total": len(all_ids)}, f)


def log_stats() -> dict:
    if os.path.exists(AUTOBACKTEST_LOG):
        with open(AUTOBACKTEST_LOG) as f:
            d = json.load(f)
        return {"total": d.get("total", 0)}
    return {"total": 0}


# ── data loading ──────────────────────────────────────────────────────────────

def _preload_team_logs(status_cb=None) -> dict:
    """
    Fetch game logs for all 30 NBA teams in one pass.
    Uses SQLite cache so repeat calls are instant — only hits the API for
    teams not yet cached.  Returns {team_id (int): sorted DataFrame}.
    """
    def report(msg):
        if status_cb: status_cb(msg)

    all_teams = nba_teams_static.get_teams()
    team_logs = {}
    fresh     = 0

    for team in all_teams:
        tid       = team["id"]
        cache_key = f"lgf_{tid}"
        df        = cache.get(cache_key)

        if df is None:
            try:
                finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=tid)
                df     = finder.get_data_frames()[0]
                cache.set(cache_key, df)
                fresh += 1
                time.sleep(0.5)
            except Exception:
                continue

        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        team_logs[tid]  = df.sort_values("GAME_DATE").reset_index(drop=True)

    report(
        f"Team logs ready — {len(team_logs)} teams "
        f"({fresh} fresh API calls, {len(team_logs) - fresh} from cache)."
    )
    return team_logs


def _fetch_season_games(status_cb=None) -> pd.DataFrame:
    """Fetch all completed season games as paired home/away rows."""
    def report(msg):
        if status_cb: status_cb(msg)

    all_games = []
    for s in SEASONS:
        try:
            report(f"Fetching {s} schedule…")
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
        "TEAM_ID_h": "HOME_TEAM_ID",
        "TEAM_ID_a": "AWAY_TEAM_ID",
        "WL_h":      "WL_HOME",
    })
    paired["HOME_WON"] = (paired["WL_HOME"] == "W").astype(int)
    return (
        paired[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_WON"]]
        .sort_values("GAME_DATE")
        .reset_index(drop=True)
    )


# ── in-memory stat computation (replaces per-game API calls) ─────────────────

def _stats_from_log(df: pd.DataFrame, team_name: str, target_date,
                    window: int = 7) -> dict | None:
    """
    Compute rolling pre-game stats from a pre-loaded team DataFrame.
    Mirrors get_team_advanced_stats() but uses no API or cache at all.
    """
    needed = ["PLUS_MINUS", "PTS", "FGM", "FGA", "FG3M",
              "FTA", "AST", "TOV", "REB", "OREB", "DREB", "WL"]
    df = df[df["GAME_DATE"] < pd.to_datetime(target_date)].dropna(subset=needed)
    if len(df) < 3:
        return None

    recent = df.tail(window).copy()
    recent["PTS_ALLOWED"] = recent["PTS"] - recent["PLUS_MINUS"]

    tot_fga  = recent["FGA"].sum()
    efg_pct  = (recent["FGM"].sum() + 0.5 * recent["FG3M"].sum()) / tot_fga if tot_fga > 0 else 0
    ast_tov  = recent["AST"].sum() / max(recent["TOV"].sum(), 1)
    tot_pts  = recent["PTS"].sum()
    ts_den   = 2 * (tot_fga + 0.44 * recent["FTA"].sum())
    ts_pct   = tot_pts / ts_den if ts_den > 0 else 0
    tot_oreb = recent["OREB"].sum()
    tot_dreb = recent["DREB"].sum()
    oreb_pct = tot_oreb / (tot_oreb + tot_dreb) if (tot_oreb + tot_dreb) > 0 else 0

    streak = 0
    for wl in reversed(recent["WL"].tolist()):
        if wl == "W":
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1

    last_game = df["GAME_DATE"].iloc[-1]
    days_rest = min(int((pd.to_datetime(target_date) - last_game).days), 10)

    avg_pts     = recent["PTS"].mean()
    avg_pts_all = recent["PTS_ALLOWED"].mean()
    std_pts     = max(recent["PTS"].std() if len(recent) > 1 else 6.0, 3.0)
    std_pts_all = max(recent["PTS_ALLOWED"].std() if len(recent) > 1 else 6.0, 3.0)

    return {
        "name":            team_name,
        "team_id":         int(df["TEAM_ID"].iloc[-1]),
        "PLUS_MINUS":      recent["PLUS_MINUS"].mean(),
        "REB":             recent["REB"].mean(),
        "eFG_PCT":         efg_pct,
        "AST_TOV":         ast_tov,
        "TS_PCT":          ts_pct,
        "WIN_STREAK":      streak,
        "OREB_PCT":        oreb_pct,
        "DAYS_REST":       days_rest,
        "IS_B2B":          1 if days_rest <= 1 else 0,
        "PTS":             avg_pts,
        "PTS_STD":         std_pts,
        "PTS_ALLOWED":     avg_pts_all,
        "PTS_ALLOWED_STD": std_pts_all,
        "ELO":             get_team_elo(team_name),
    }


# ── core prediction loop ─────────────────────────────────────────────────────

def _core_backtest(games: pd.DataFrame, team_logs: dict,
                   model, status_cb=None) -> dict:
    """
    Blindfold-predict every row in `games` using pre-loaded team logs.
    Pure in-memory — zero API calls after team logs are loaded.
    """
    def report(msg):
        if status_cb: status_cb(msg)

    total         = len(games)
    correct       = 0
    lessons       = []
    processed_ids = set()

    for idx, row in games.iterrows():
        game_date = row["GAME_DATE"]
        home_tid  = int(row["HOME_TEAM_ID"])
        away_tid  = int(row["AWAY_TEAM_ID"])
        home_name = _TEAM_MAP.get(home_tid)
        away_name = _TEAM_MAP.get(away_tid)
        h_log     = team_logs.get(home_tid)
        a_log     = team_logs.get(away_tid)

        if not home_name or not away_name or h_log is None or a_log is None:
            continue

        hd = _stats_from_log(h_log, home_name, game_date)
        ad = _stats_from_log(a_log, away_name, game_date)

        if not hd or not ad:
            continue

        try:
            fp, *_  = predict_hybrid(hd, ad, model)
            pred    = 1 if fp > 0.5 else 0
            actual  = int(row["HOME_WON"])

            if pred == actual:
                correct += 1
            else:
                for is_home, stats, result in [(1, hd, actual), (0, ad, 1 - actual)]:
                    lesson = {
                        "GAME_DATE":  str(game_date.date()), "TEAM_ID": 0,
                        "IS_HOME":    is_home,
                        "PLUS_MINUS": stats["PLUS_MINUS"],
                        "PTS":        stats["PTS"],        "REB":        stats["REB"],
                        "eFG_PCT":    stats["eFG_PCT"],    "AST_TOV":    stats["AST_TOV"],
                        "TS_PCT":     stats["TS_PCT"],     "WIN_STREAK": stats["WIN_STREAK"],
                        "OREB_PCT":   stats["OREB_PCT"],   "DAYS_REST":  stats["DAYS_REST"],
                        "IS_B2B":     stats["IS_B2B"],     "RESULT":     result,
                    }
                    lessons.extend([lesson] * 3)

            processed_ids.add(row["GAME_ID"])
        except Exception:
            continue

        done = len(processed_ids)
        if done % 30 == 0 and done > 0:
            acc_so_far = correct / done * 100
            report(f"  {done}/{total} games · {acc_so_far:.1f}% correct so far…")

    return {"correct": correct, "lessons": lessons, "processed_ids": processed_ids}


def _apply_lessons_and_retrain(lessons: list, status_cb=None) -> dict | None:
    def report(msg):
        if status_cb: status_cb(msg)
    if not lessons or not os.path.exists(DATASET_FILE):
        return None
    report(f"Injecting {len(lessons) // 6} wrong-prediction lessons into dataset…")
    master  = pd.read_csv(DATASET_FILE)
    updated = pd.concat([master, pd.DataFrame(lessons)], ignore_index=True)
    updated.to_csv(DATASET_FILE, index=False)
    report("Retraining model on updated dataset…")
    _, _, metrics = train_model(force_rebuild=False, status_cb=status_cb)
    return metrics


# ── public entry points ───────────────────────────────────────────────────────

def run_new_games(model, status_cb=None) -> dict:
    """
    Incremental backtest: only processes games not yet in the log.
    Use this daily to keep the model up to date.
    """
    def report(msg):
        if status_cb: status_cb(msg)

    all_games = _fetch_season_games(status_cb=status_cb)
    if all_games.empty:
        return {"new": 0, "correct": 0, "lessons": 0, "acc": 0.0, "metrics": None}

    new_games = all_games[~all_games["GAME_ID"].isin(load_log())].reset_index(drop=True)
    if new_games.empty:
        report("No new games since last run.")
        return {"new": 0, "correct": 0, "lessons": 0, "acc": 0.0, "metrics": None}

    report(f"{len(new_games)} new games found. Pre-loading all team logs…")
    team_logs = _preload_team_logs(status_cb=status_cb)

    res     = _core_backtest(new_games, team_logs, model, status_cb=status_cb)
    n       = len(res["processed_ids"])
    acc     = res["correct"] / n if n > 0 else 0.0
    metrics = _apply_lessons_and_retrain(res["lessons"], status_cb=status_cb)
    _save_log(res["processed_ids"])

    report(f"Done — {n} new games · {res['correct']}/{n} correct ({acc * 100:.1f}%)")
    return {"new": n, "correct": res["correct"],
            "lessons": len(res["lessons"]) // 6, "acc": acc, "metrics": metrics}


def run_full_season(model, status_cb=None) -> dict:
    """
    Full season blind run: blindfold-predict EVERY completed game this season
    so the model learns from the whole history at once.
    Run this once to build up the model's memory, then use run_new_games()
    for daily incremental updates.
    """
    def report(msg):
        if status_cb: status_cb(msg)

    report("Starting full season blind training run…")
    all_games = _fetch_season_games(status_cb=status_cb)
    if all_games.empty:
        return {"new": 0, "correct": 0, "lessons": 0, "acc": 0.0, "metrics": None}

    report(f"{len(all_games)} completed games found. Pre-loading all team logs…")
    team_logs = _preload_team_logs(status_cb=status_cb)

    res     = _core_backtest(all_games, team_logs, model, status_cb=status_cb)
    n       = len(res["processed_ids"])
    acc     = res["correct"] / n if n > 0 else 0.0
    metrics = _apply_lessons_and_retrain(res["lessons"], status_cb=status_cb)
    _save_log(res["processed_ids"])

    report(f"Full run complete — {n} games · {res['correct']}/{n} correct ({acc * 100:.1f}%)")
    return {"new": n, "correct": res["correct"],
            "lessons": len(res["lessons"]) // 6, "acc": acc, "metrics": metrics}
