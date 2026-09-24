"""nflreadpy loading with a local parquet cache, plus game / play lookup."""

from __future__ import annotations

import os
import re
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import polars as pl

from .teams import franchise

FIRST_SEASON, LAST_SEASON = 2010, 2019
# Schedules reach further back (Elo warm-up) and forward (two seasons after 2019).
SCHEDULE_SEASONS = range(2002, 2022)

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
    "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100", "goal_to_go", "time",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "posteam_score", "defteam_score", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "play_type", "desc", "yards_gained", "air_yards", "return_yards",
    "interception", "fumble_lost", "return_touchdown", "touchdown", "td_team",
    "incomplete_pass", "out_of_bounds", "sack", "qb_spike", "qb_kneel", "timeout", "timeout_team",
    "penalty", "penalty_team", "penalty_yards", "first_down_penalty",
    "two_point_attempt", "two_point_conv_result", "extra_point_result",
    "field_goal_result", "kick_distance", "safety",
    "wp", "wpa", "home_wp", "home_wp_post", "ep", "epa",
    "fixed_drive", "fixed_drive_result", "home_opening_kickoff",
    "total_home_score", "total_away_score", "result",
]
TEAM_COLUMNS = ["home_team", "away_team", "posteam", "defteam", "td_team", "timeout_team", "penalty_team"]


def cache_dir() -> Path:
    d = Path(os.environ.get("WHATIF_CACHE", Path(__file__).resolve().parent.parent / "data_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _normalize_teams(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = df[c].map(franchise, na_action="ignore")
    return df


def load_pbp_season(season: int, refresh: bool = False) -> pd.DataFrame:
    path = cache_dir() / f"pbp_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = nfl.load_pbp([season])
    cols = [c for c in PBP_COLUMNS if c in raw.columns]
    df = raw.select(cols).to_pandas()
    df = _normalize_teams(df, TEAM_COLUMNS).sort_values(["game_id", "play_id"]).reset_index(drop=True)
    df.to_parquet(path)
    return df


def load_pbp(seasons, refresh: bool = False) -> pd.DataFrame:
    return pd.concat([load_pbp_season(s, refresh) for s in seasons], ignore_index=True)


def load_schedules(refresh: bool = False) -> pd.DataFrame:
    path = cache_dir() / "schedules.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = nfl.load_schedules(list(SCHEDULE_SEASONS))
    cols = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "away_score",
            "home_team", "home_score", "location", "result", "overtime", "div_game", "spread_line"]
    df = raw.select(cols).filter(pl.col("game_type").is_not_null()).to_pandas()
    df = _normalize_teams(df, ["home_team", "away_team"])
    df = df.sort_values(["season", "gameday", "gametime", "game_id"]).reset_index(drop=True)
    df.to_parquet(path)
    return df


# ---- lookup ---------------------------------------------------------------------
def find_games(schedules: pd.DataFrame, season: int | None = None, week: int | None = None,
               team: str | None = None, game_type: str | None = None) -> pd.DataFrame:
    g = schedules
    if season is not None:
        g = g[g["season"] == season]
    if week is not None:
        g = g[g["week"] == week]
    if team:
        g = g[(g["home_team"] == team) | (g["away_team"] == team)]
    if game_type:
        g = g[g["game_type"] == game_type.upper()]
    return g


def load_game(game_id: str, refresh: bool = False) -> pd.DataFrame:
    season = int(game_id[:4])
    pbp = load_pbp_season(season, refresh)
    game = pbp[pbp["game_id"] == game_id].reset_index(drop=True)
    if game.empty:
        raise ValueError(f"no play-by-play for game {game_id!r}")
    return game


def parse_clock(text: str) -> float:
    """'0:26' / ':26' / '12:05' -> seconds."""
    m = re.fullmatch(r"\s*(\d*):(\d{1,2})\s*", text)
    if not m:
        raise ValueError(f"clock must look like 12:34 or :26, got {text!r}")
    return int(m.group(1) or 0) * 60 + int(m.group(2))


def find_plays(game: pd.DataFrame, qtr: int | None = None, clock: str | None = None,
               search: str | None = None, min_abs_wpa: float | None = None,
               clock_window: float = 30) -> pd.DataFrame:
    """Filter a game's snaps. `clock` matches within `clock_window` seconds."""
    p = game[game["play_type"].notna() & game["posteam"].notna()]
    if qtr is not None:
        p = p[p["qtr"] == qtr]
    if clock:
        secs = parse_clock(clock)
        p = p[(p["quarter_seconds_remaining"] - secs).abs() <= clock_window]
    if search:
        p = p[p["desc"].str.contains(search, case=False, regex=False, na=False)]
    if min_abs_wpa is not None:
        p = p[p["wpa"].abs() >= min_abs_wpa]
    return p


def play_summary(p: pd.DataFrame) -> pd.DataFrame:
    """Compact table for listing plays."""
    out = pd.DataFrame({
        "play_id": p["play_id"].astype(int),
        "qtr": p["qtr"].astype(int),
        "clock": p["time"],
        "team": p["posteam"],
        "down": p["down"].map(lambda d: "" if pd.isna(d) else f"{int(d)}"),
        "togo": p["ydstogo"].map(lambda d: "" if pd.isna(d) else f"{int(d)}"),
        "yardline": p["yardline_100"].map(lambda d: "" if pd.isna(d) else f"{int(d)}"),
        "score": p.apply(lambda r: f"{int(r.posteam_score)}-{int(r.defteam_score)}"
                         if pd.notna(r.posteam_score) else "", axis=1),
        "wpa": p["wpa"].round(3),
        "desc": p["desc"],
    })
    return out.reset_index(drop=True)
