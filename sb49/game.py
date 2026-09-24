"""Turn real play-by-play rows into GameStates, apply a changed result to one
play, and estimate win probability by Monte Carlo."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import SNAP_TYPES, LeagueModel
from .simulate import Engine, GameOver
from .state import KICKOFF, PAT, SCRIMMAGE, GameState

OUTCOMES = {
    "touchdown": "offense scores a touchdown",
    "gain": "run/completion for --yards (negative = loss); handles first downs, TDs, safeties, turnover on downs",
    "incomplete": "incomplete pass",
    "interception": "defense intercepts; --yards = spot of the change of possession past the line of scrimmage",
    "fumble": "offense fumbles and loses it; --yards as for interception",
    "defensive_td": "turnover returned for a touchdown",
    "field_goal_made": "field goal is good",
    "field_goal_missed": "field goal is no good",
    "punt": "punt with --yards net",
    "safety": "offense is tackled in its own end zone",
    "kneel": "quarterback kneel (-1 yard)",
}


# ---- real game -> states ----------------------------------------------------------
def state_from_row(row: pd.Series) -> GameState | None:
    """Pre-snap GameState for a real play, or None for non-snap rows."""
    if row["play_type"] not in SNAP_TYPES or pd.isna(row["posteam"]):
        return None
    home, away = row["home_team"], row["away_team"]
    pos, dfn = row["posteam"], row["defteam"]
    score = {pos: int(row["posteam_score"]), dfn: int(row["defteam_score"])}
    timeouts = {pos: int(row["posteam_timeouts_remaining"]), dfn: int(row["defteam_timeouts_remaining"])}
    qtr = int(row["qtr"])
    half = 1 if qtr <= 2 else (2 if qtr <= 4 else 3)
    receiver_2h = away if row["home_opening_kickoff"] == 1 else home
    s = GameState(
        offense=pos, defense=dfn, half=half, secs=float(row["half_seconds_remaining"]),
        score=score, timeouts=timeouts, second_half_receiver=receiver_2h,
    )
    if row["play_type"] == "kickoff":
        s.swap()  # nflfastR lists the receiving team as posteam on kickoffs
        s.phase = KICKOFF
    elif row["play_type"] == "extra_point" or row["two_point_attempt"] == 1:
        s.phase = PAT
    else:
        s.phase = SCRIMMAGE
        s.yardline_100 = int(row["yardline_100"])
        s.down = int(row["down"])
        s.ydstogo = int(row["ydstogo"])
    return s


def real_duration(game: pd.DataFrame, idx: int) -> float:
    """Clock run-off of a real play (until the next snap)."""
    here = game.loc[idx, "game_seconds_remaining"]
    later = game.loc[idx + 1:]
    later = later[later["play_type"].isin(SNAP_TYPES)]
    if later.empty:
        return here
    return float(max(0.0, here - later.iloc[0]["game_seconds_remaining"]))


def play_row(game: pd.DataFrame, play_id: int) -> int:
    matches = game.index[game["play_id"] == play_id]
    if len(matches) == 0:
        raise ValueError(f"play_id {play_id} is not in this game; run with --list to see play ids")
    return int(matches[0])


# ---- override -------------------------------------------------------------------------
@dataclass
class Override:
    outcome: str
    yards: int | None = None
    clock_secs: float | None = None


def apply_override(state: GameState, ov: Override, default_secs: float, model: LeagueModel) -> tuple[GameState, str | None]:
    """Return (post-play state, winner if the game ended on this play)."""
    if state.phase != SCRIMMAGE:
        raise ValueError("only scrimmage plays (runs, passes, punts, field goals, kneels) can be changed, not kickoffs or tries after TDs")
    if ov.outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {ov.outcome!r}; choose from {sorted(OUTCOMES)}")
    s = state.copy()
    eng = Engine(model, random.Random(0))
    stopping = {"incomplete": 5, "touchdown": 6, "interception": 6, "fumble": 6, "defensive_td": 8,
                "field_goal_made": 5, "field_goal_missed": 5, "safety": 6}
    secs = ov.clock_secs if ov.clock_secs is not None else stopping.get(ov.outcome, default_secs)
    y = ov.yards
    try:
        eng.run_clock(s, secs)
        match ov.outcome:
            case "touchdown":
                eng.touchdown(s, s.offense)
            case "gain":
                eng.gain(s, _need(y, "gain"))
            case "incomplete":
                eng.gain(s, 0)
            case "interception" | "fumble":
                eng.turnover(s, y or 0)
            case "defensive_td":
                eng.turnover(s, 0, defense_td=True)
            case "field_goal_made":
                eng.field_goal(s, True)
            case "field_goal_missed":
                eng.field_goal(s, False)
            case "punt":
                eng.punt(s, _need(y, "punt"))
            case "safety":
                eng.safety(s)
            case "kneel":
                eng.gain(s, -1)
    except GameOver as end:
        return s, end.winner
    return s, None


def _need(y: int | None, outcome: str) -> int:
    if y is None:
        raise ValueError(f"outcome {outcome!r} needs --yards")
    return y


# ---- Monte Carlo ------------------------------------------------------------------------
_MODEL: LeagueModel | None = None


def _init(model: LeagueModel) -> None:
    global _MODEL
    _MODEL = model


def _simulate(args: tuple[GameState, int, int]) -> list[tuple[str, int]]:
    """Run n games from state; return (winner, final margin for the state's offense)."""
    state, n, seed = args
    eng = Engine(_MODEL, random.Random(seed))
    a, b = state.offense, state.defense
    out = []
    for _ in range(n):
        r = eng.play_out(state.copy())
        out.append((r.winner, r.score[a] - r.score[b]))
    return out


@dataclass
class MCResult:
    wins: dict[str, int]
    n: int
    margins: np.ndarray   # final margin, first team in `teams` minus second
    teams: tuple[str, str]

    def prob(self, team: str) -> float:
        return self.wins.get(team, 0) / self.n

    def ci95(self, team: str) -> tuple[float, float]:
        p = self.prob(team)
        half = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / self.n)
        return max(0.0, p - half), min(1.0, p + half)


def _pool(model: LeagueModel, workers: int | None):
    workers = workers or os.cpu_count() or 1
    return mp.get_context("fork").Pool(workers, initializer=_init, initargs=(model,))


def monte_carlo(state: GameState, model: LeagueModel, n: int, seed: int = 1, workers: int | None = None) -> MCResult:
    """Win probability from a single state, split across processes."""
    workers = workers or os.cpu_count() or 1
    chunks = [(state, n // workers + (i < n % workers), seed * 1000 + i) for i in range(workers)]
    with _pool(model, workers) as pool:
        results = [r for part in pool.map(_simulate, chunks) for r in part]
    return _collect(state, results)


def monte_carlo_many(states: list[GameState], model: LeagueModel, n: int, seed: int = 1,
                     workers: int | None = None) -> list[MCResult]:
    """Win probability for many states (one task per state)."""
    with _pool(model, workers) as pool:
        parts = pool.map(_simulate, [(s, n, seed * 100_000 + i) for i, s in enumerate(states)], chunksize=1)
    return [_collect(s, r) for s, r in zip(states, parts)]


def _collect(state: GameState, results: list[tuple[str, int]]) -> MCResult:
    wins: dict[str, int] = {}
    for w, _ in results:
        wins[w] = wins.get(w, 0) + 1
    return MCResult(wins, len(results), np.array([m for _, m in results]), (state.offense, state.defense))
