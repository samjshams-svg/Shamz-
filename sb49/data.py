"""Load Super Bowl XLIX and league-wide 2014 play-by-play via nflreadpy."""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import pandas as pd

GAME_ID = "2014_21_NE_SEA"
SEASON = 2014

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

# Only the columns the model and the game replay need.
COLUMNS = [
    "game_id", "play_id", "season_type", "home_team", "away_team",
    "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100",
    "game_seconds_remaining", "half_seconds_remaining",
    "posteam_score", "defteam_score", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "play_type", "desc", "yards_gained", "air_yards", "return_yards",
    "interception", "fumble_lost", "return_touchdown", "touchdown", "td_team",
    "incomplete_pass", "out_of_bounds", "qb_spike", "qb_kneel", "timeout",
    "two_point_attempt", "two_point_conv_result", "extra_point_result",
    "field_goal_result", "kick_distance", "touchback", "safety",
    "home_wp", "home_wp_post", "home_opening_kickoff", "aborted_play",
]


def load_season(season: int = SEASON, use_cache: bool = True) -> pd.DataFrame:
    """Full-season play-by-play (regular season + playoffs) as pandas."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"pbp_{season}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    pbp = nfl.load_pbp([season]).select(COLUMNS).to_pandas()
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    if use_cache:
        pbp.to_parquet(cache)
    return pbp


def load_game(pbp: pd.DataFrame, game_id: str = GAME_ID) -> pd.DataFrame:
    game = pbp[pbp["game_id"] == game_id].sort_values("play_id").reset_index(drop=True)
    if game.empty:
        raise ValueError(f"game {game_id!r} not found in play-by-play")
    return game
