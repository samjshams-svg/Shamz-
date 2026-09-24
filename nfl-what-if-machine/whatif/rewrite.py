"""Part 1: rewrite one play and Monte Carlo the rest of the game."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from .gamesim import Engine, GameOver
from .models import SNAP_TYPES, GameModel, load_model
from .state import KICKOFF, PAT, SCRIMMAGE, GameState, ot_length
from .teams import resolve_team

OUTCOMES = {
    "run": "call a run instead; its result is drawn from real runs in the same situation",
    "pass": "call a pass instead; its result is drawn from real passes in the same situation",
    "touchdown": "offense scores a touchdown",
    "gain": "run or catch for --yards (negative = loss)",
    "incomplete": "incomplete pass",
    "sack": "sack for --yards lost (default 7)",
    "interception": "intercepted; --yards = where the defense ends up past the line (default 0)",
    "fumble_lost": "fumble recovered by the defense; --yards as for interception",
    "defensive_td": "turnover returned for a touchdown",
    "field_goal_made": "field goal is good",
    "field_goal_missed": "field goal is no good",
    "punt": "punt with --yards net (default 40)",
    "safety": "offense tackled in its own end zone",
    "kneel": "quarterback kneel",
    "spike": "spike to stop the clock",
    "penalty": "accepted penalty of --yards on --penalty-on offense/defense",
    "custom": "no play result: start from the state you set with the override options",
}
CLOCK_STOPPING = {"incomplete": 5, "spike": 2, "touchdown": 6, "interception": 6, "fumble_lost": 6,
                  "defensive_td": 8, "field_goal_made": 5, "field_goal_missed": 5, "safety": 6,
                  "punt": 8, "penalty": 5, "custom": 0}


@dataclass
class Edit:
    outcome: str
    yards: int | None = None
    penalty_on: str = "defense"
    automatic_first_down: bool = False
    clock_secs: float | None = None       # run-off of the edited play
    # State overrides (applied after the play; before it for run / pass / custom).
    possession: str | None = None
    down: int | None = None
    ydstogo: int | None = None
    yardline: str | int | None = None     # "NE 1", "own 20", "opp 35", "50", or yards to the opponent goal
    score: dict[str, int] | None = None
    clock: str | None = None              # "Q4 0:20"
    timeouts: dict[str, int] | None = None

    def describe(self) -> str:
        text = self.outcome.replace("_", " ")
        if self.yards is not None and self.outcome in ("gain", "sack", "punt", "penalty", "interception", "fumble_lost"):
            text += f" ({self.yards:+d} yds)" if self.outcome == "gain" else f" ({self.yards} yds)"
        if self.outcome == "penalty":
            text += f" on {self.penalty_on}"
        return text


# ---- real rows -> states ----------------------------------------------------------
def state_from_row(row: pd.Series, game: pd.DataFrame | None = None) -> GameState | None:
    """Pre-snap state of a real play (None for rows that aren't snaps)."""
    if row["play_type"] not in SNAP_TYPES or pd.isna(row["posteam"]) or pd.isna(row["posteam_score"]):
        return None
    pos, dfn = row["posteam"], row["defteam"]
    qtr = int(row["qtr"])
    half = 1 if qtr <= 2 else 2 if qtr <= 4 else qtr - 2
    s = GameState(
        offense=pos, defense=dfn, home=row["home_team"], away=row["away_team"],
        season=int(row["season"]), playoff=row["season_type"] == "POST", half=half,
        secs=float(row["half_seconds_remaining"] if qtr <= 4 else row["quarter_seconds_remaining"]),
        score={pos: int(row["posteam_score"]), dfn: int(row["defteam_score"])},
        timeouts={pos: int(row["posteam_timeouts_remaining"]), dfn: int(row["defteam_timeouts_remaining"])},
        second_half_receiver=row["away_team"] if row["home_opening_kickoff"] == 1 else row["home_team"],
    )
    if half > 2 and game is not None:
        s.ot_stage = _ot_stage(game, row)
    if row["play_type"] == "kickoff":
        s.swap()  # nflfastR lists the receiving team as posteam on kickoffs
        s.phase, s.mid_drive = KICKOFF, False
    elif row["play_type"] == "extra_point" or row["two_point_attempt"] == 1:
        s.phase, s.mid_drive = PAT, False
    else:
        s.phase = SCRIMMAGE
        s.yardline_100, s.down, s.ydstogo = int(row["yardline_100"]), int(row["down"]), int(row["ydstogo"])
    return s


def _ot_stage(game: pd.DataFrame, row: pd.Series) -> int:
    ot = game[(game["qtr"] >= 5) & (game["play_id"] < row["play_id"]) & game["posteam"].notna()
              & game["down"].notna()]
    if ot.empty or ot["posteam"].nunique() == 1 and ot["posteam"].iloc[0] == row["posteam"]:
        return 0
    return 2 if row["posteam_score"] == row["defteam_score"] else 1


def real_runoff(game: pd.DataFrame, idx: int) -> float:
    here = game.loc[idx, "game_seconds_remaining"]
    later = game.loc[idx + 1:]
    later = later[later["play_type"].isin(SNAP_TYPES)]
    if later.empty or pd.isna(here):
        return 0.0
    return float(max(0.0, here - later.iloc[0]["game_seconds_remaining"]))


def next_snap_state(game: pd.DataFrame, idx: int) -> GameState | None:
    """State after the real play: the next snap's pre-snap state."""
    for j in range(idx + 1, len(game)):
        s = state_from_row(game.loc[j], game)
        if s is not None:
            return s
    return None


# ---- parsing overrides ------------------------------------------------------------
def parse_yardline(text: str | int, offense: str, defense: str) -> int:
    """'NE 1' / 'own 20' / 'opp 35' / '50' / 1 -> yards from the offense's goal line."""
    if isinstance(text, (int, np.integer)):
        return int(text)
    t = str(text).strip()
    if re.fullmatch(r"\d+", t):
        return int(t)
    m = re.fullmatch(r"(\w+)\s+(\d+)", t)
    if not m:
        raise ValueError(f"yard line must look like 'NE 1', 'own 20' or 'opp 35', got {text!r}")
    side, n = m.group(1).lower(), int(m.group(2))
    if side in ("own",) or (side not in ("opp",) and resolve_team(side) == offense):
        return 100 - n
    if side == "opp" or resolve_team(side) == defense:
        return n
    raise ValueError(f"{m.group(1)} is not in this game")


def parse_game_clock(text: str) -> tuple[int, float]:
    """'Q4 0:20' / '4 :20' / 'OT 3:00' -> (quarter, seconds left in the quarter)."""
    m = re.fullmatch(r"\s*(?:Q?(\d)|(OT)(\d?))\s+(\d*:\d{1,2})\s*", text, flags=re.I)
    if not m:
        raise ValueError(f"clock must look like 'Q4 0:20' or 'OT 3:00', got {text!r}")
    qtr = int(m.group(1)) if m.group(1) else 4 + int(m.group(3) or 1)
    return qtr, data.parse_clock(m.group(4))


def _apply_overrides(s: GameState, e: Edit) -> None:
    if e.possession:
        team = resolve_team(e.possession)
        if team not in (s.offense, s.defense):
            raise ValueError(f"{team} is not playing in this game")
        if team != s.offense:
            s.swap()
    if e.score:
        for team, pts in e.score.items():
            s.score[resolve_team(team)] = int(pts)
    if e.timeouts:
        for team, n in e.timeouts.items():
            s.timeouts[resolve_team(team)] = int(n)
    if e.clock:
        qtr, qsecs = parse_game_clock(e.clock)
        if qtr <= 4:
            s.half = 1 if qtr <= 2 else 2
            s.secs = qsecs + (900 if qtr in (1, 3) else 0)
        else:
            s.half, s.secs = qtr - 2, min(qsecs, ot_length(s.season, s.playoff))
    if e.yardline is not None:
        s.yardline_100 = parse_yardline(e.yardline, s.offense, s.defense)
    if e.down is not None:
        s.down = int(e.down)
    if e.ydstogo is not None:
        s.ydstogo = int(e.ydstogo)
    if any(v is not None for v in (e.possession, e.down, e.ydstogo, e.yardline)):
        s.phase, s.mid_drive = SCRIMMAGE, True
    if not 1 <= s.yardline_100 <= 99:
        raise ValueError("yard line must be between 1 and 99 yards from the goal")
    if not 1 <= s.down <= 4:
        raise ValueError("down must be 1-4")
    s.ydstogo = max(1, min(s.ydstogo, s.yardline_100))


def apply_edit(pre: GameState, e: Edit, model: GameModel, runoff: float) -> tuple[GameState, str | None, bool]:
    """Return (state to simulate from, winner if the edit ended the game, game_over)."""
    if e.outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {e.outcome!r}; choose from {', '.join(OUTCOMES)}")
    if pre.phase != SCRIMMAGE and e.outcome not in ("custom",):
        raise ValueError("only snaps from scrimmage can be edited (not kickoffs or tries after TDs); use 'custom'")
    s = pre.copy()
    s.mid_drive = True
    if e.outcome in ("run", "pass", "custom"):
        _apply_overrides(s, e)
        if e.outcome != "custom":
            s.forced_call = e.outcome
        return s, None, False

    eng = Engine(model, random.Random(0))
    secs = e.clock_secs if e.clock_secs is not None else CLOCK_STOPPING.get(e.outcome, runoff)
    y = e.yards
    try:
        eng.run_clock(s, secs)
        match e.outcome:
            case "touchdown":
                eng.touchdown(s, s.offense)
            case "gain":
                eng.gain(s, _need(y, "gain"))
            case "incomplete" | "spike":
                eng.gain(s, 0)
            case "sack":
                eng.gain(s, -abs(y if y is not None else 7))
            case "interception" | "fumble_lost":
                eng.turnover(s, y or 0)
            case "defensive_td":
                eng.turnover(s, 0, defense_td=True)
            case "field_goal_made":
                eng.field_goal(s, True)
            case "field_goal_missed":
                eng.field_goal(s, False)
            case "punt":
                eng.punt(s, y if y is not None else 40)
            case "safety":
                eng.safety(s)
            case "kneel":
                eng.gain(s, -1)
            case "penalty":
                eng.penalty(s, e.penalty_on.lower().startswith("off"), abs(_need(y, "penalty")),
                            e.automatic_first_down)
    except GameOver as end:
        return s, end.winner, True
    _apply_overrides(s, e)
    return s, None, False


def _need(y: int | None, outcome: str) -> int:
    if y is None:
        raise ValueError(f"outcome {outcome!r} needs a yards value")
    return y


# ---- Monte Carlo ------------------------------------------------------------------------
_MODEL: GameModel | None = None


def _init(model: GameModel) -> None:
    global _MODEL
    _MODEL = model


def _simulate(args: tuple[GameState, int, int, bool]):
    state, n, seed, record = args
    eng = Engine(_MODEL, random.Random(seed))
    out = []
    for _ in range(n):
        r = eng.play_out(state.copy(), record=record)
        out.append((r.winner, r.score[state.home] - r.score[state.away], r.path))
    return out


@dataclass
class GameMC:
    home: str
    away: str
    n: int
    home_wins: int
    away_wins: int
    ties: int
    margins: np.ndarray                    # home minus away, one per simulation
    paths: list = field(default_factory=list)     # per simulation: [(minutes, situation), ...]
    home_outcome: np.ndarray | None = None        # 1 home win, 0 loss, 0.5 tie

    def prob(self, team: str) -> float:
        return (self.home_wins if team == self.home else self.away_wins) / self.n

    @property
    def tie_prob(self) -> float:
        return self.ties / self.n

    def ci95(self, team: str) -> tuple[float, float]:
        p = self.prob(team)
        h = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / self.n)
        return max(0.0, p - h), min(1.0, p + h)


def monte_carlo(state: GameState, model: GameModel, n: int = 10_000, seed: int = 1,
                workers: int | None = None, record: bool = False) -> GameMC:
    """`record` keeps every simulated path's situations (for the win-probability chart)."""
    workers = max(1, min(workers or os.cpu_count() or 1, n))
    sizes = [n // workers + (i < n % workers) for i in range(workers)]
    jobs = [(state, sizes[i], seed * 1000 + i, record) for i in range(workers)]
    if workers == 1:
        _init(model)
        parts = [_simulate(jobs[0])]
    else:
        with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(model,)) as pool:
            parts = pool.map(_simulate, jobs)
    rows = [r for part in parts for r in part]
    return _collect(state, rows)


def fixed_result(state: GameState, winner: str | None, n: int) -> GameMC:
    """An edit that ended the game: every 'simulation' is the same final."""
    margin = state.score[state.home] - state.score[state.away]
    return GameMC(state.home, state.away, n, n * (winner == state.home), n * (winner == state.away),
                  n * (winner is None), np.full(n, margin), [])


def _collect(state: GameState, rows) -> GameMC:
    hw = sum(1 for w, _, _ in rows if w == state.home)
    aw = sum(1 for w, _, _ in rows if w == state.away)
    outcome = np.array([0.5 if w is None else float(w == state.home) for w, _, _ in rows])
    return GameMC(state.home, state.away, len(rows), hw, aw, len(rows) - hw - aw,
                  np.array([m for _, m, _ in rows]), [p for _, _, p in rows if p], outcome)


# ---- the whole Part 1 pipeline --------------------------------------------------------------
@dataclass
class GameRewrite:
    game_id: str
    home: str
    away: str
    play_id: int
    play_desc: str
    edit: Edit
    pre: GameState
    post: GameState
    alt: GameMC                        # after the edited play
    real_after: GameMC | None          # our model, after what really happened
    model_before: GameMC | None        # our model, before the snap (play called normally)
    nflfastr_before: float             # home WP
    nflfastr_after: float              # home WP
    actual_wp: pd.DataFrame            # elapsed minutes, home WP (nflfastR)
    final_score: dict[str, int]
    play_minutes: float
    ended_on_play: bool = False

    def summary(self) -> str:
        lines = [f"{self.away} at {self.home} ({self.game_id}), play {self.play_id}",
                 f"  real play : {self.play_desc}",
                 f"  before    : {self.pre.describe()}",
                 f"  edit      : {self.edit.describe()}",
                 f"  after     : {'game over' if self.ended_on_play else self.post.describe()}",
                 "", f"Win probability after the edited play ({self.alt.n:,} simulations):"]
        for t in (self.home, self.away):
            lo, hi = self.alt.ci95(t)
            lines.append(f"  {t:<4} {self.alt.prob(t):6.1%}  (95% CI {lo:.1%}-{hi:.1%})")
        if self.alt.ties:
            lines.append(f"  tie  {self.alt.tie_prob:6.1%}")
        lines.append("")
        lines.append(f"For comparison ({self.home} win probability):")
        lines.append(f"  nflfastR before the snap {self.nflfastr_before:6.1%} | after the real play {self.nflfastr_after:6.1%}")
        if self.model_before and self.real_after:
            lines.append(f"  this model before snap   {self.model_before.prob(self.home):6.1%} | after the real play "
                         f"{self.real_after.prob(self.home):6.1%}")
        fs = self.final_score
        lines.append(f"  real final: {self.away} {fs[self.away]}, {self.home} {fs[self.home]}")
        return "\n".join(lines)


def rewrite_game(game_id: str, play_id: int, edit: Edit, n: int = 10_000, seed: int = 1,
                 workers: int | None = None, compare: bool = True, record: bool = True,
                 model: GameModel | None = None) -> GameRewrite:
    game = data.load_game(game_id)
    matches = game.index[game["play_id"] == play_id]
    if len(matches) == 0:
        raise ValueError(f"play {play_id} is not in {game_id}")
    idx = int(matches[0])
    row = game.loc[idx]
    pre = state_from_row(row, game)
    if pre is None:
        raise ValueError(f"play {play_id} is not a snap (timeout, penalty-only or period marker)")
    model = model or load_model(pre.season)
    post, winner, over = apply_edit(pre, edit, model, real_runoff(game, idx))
    if over:
        alt = fixed_result(post, winner, n)
    else:
        alt = monte_carlo(post, model, n, seed, workers, record)

    real_after = model_before = None
    if compare:
        before = pre.copy()
        model_before = monte_carlo(before, model, n, seed + 1, workers)
        nxt = next_snap_state(game, idx)
        if nxt is not None:
            real_after = monte_carlo(nxt, model, n, seed + 2, workers)

    wp_rows = game[game["home_wp"].notna()]
    actual = pd.DataFrame({
        "minutes": [_elapsed(r) for _, r in wp_rows.iterrows()],
        "home_wp": wp_rows["home_wp"].to_numpy(),
    })
    last = game.iloc[-1]
    final = {row["home_team"]: int(last["total_home_score"]), row["away_team"]: int(last["total_away_score"])}
    post_wp = row["home_wp_post"] if pd.notna(row["home_wp_post"]) else float(final[pre.home] > final[pre.away])
    return GameRewrite(
        game_id=game_id, home=pre.home, away=pre.away, play_id=play_id, play_desc=str(row["desc"]),
        edit=edit, pre=pre, post=post, alt=alt, real_after=real_after, model_before=model_before,
        nflfastr_before=float(row["home_wp"]), nflfastr_after=float(post_wp), actual_wp=actual,
        final_score=final, play_minutes=pre.elapsed_minutes, ended_on_play=over,
    )


def _elapsed(r: pd.Series) -> float:
    if r["qtr"] <= 4 or pd.isna(r["qtr"]):
        return 60 - r["game_seconds_remaining"] / 60
    length = ot_length(int(r["season"]), r["season_type"] == "POST")
    return 60 + (r["qtr"] - 5) * length / 60 + (length - r["quarter_seconds_remaining"]) / 60
