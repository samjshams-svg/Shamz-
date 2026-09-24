"""FiveThirtyEight-style NFL Elo.

* Win probability: 1 / (1 + 10^(-d/400)), d = home Elo - away Elo + home-field
  advantage (none at neutral sites); in the playoffs d is multiplied by
  `playoff_mult` (538 used 1.2: favorites win playoff games more often).
* Update: K * MOV multiplier * (result - win probability), with the 538
  margin-of-victory multiplier ln(|margin| + 1) * 2.2 / (0.001 * d_winner + 2.2),
  which damps blowouts by heavy favorites (autocorrelation correction).
* Between seasons ratings regress toward 1505 by `regress` (538: 1/3).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import data
from .teams import TEAMS

MEAN = 1505.0
PLAYOFF_TYPES = {"WC", "DIV", "CON", "SB"}


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    hfa: float = 65.0
    regress: float = 1 / 3
    playoff_mult: float = 1.2

    def label(self) -> str:
        return f"K={self.k:g}, HFA={self.hfa:g}, regression={self.regress:.2f}, playoff x{self.playoff_mult:g}"


def win_prob(diff):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff) / 400.0))


def mov_mult(margin, winner_diff):
    return np.log(np.abs(margin) + 1.0) * 2.2 / (np.asarray(winner_diff) * 0.001 + 2.2)


def game_diff(elo_home, elo_away, neutral: bool, playoff: bool, p: EloParams):
    d = np.asarray(elo_home) - np.asarray(elo_away) + (0.0 if neutral else p.hfa)
    return d * (p.playoff_mult if playoff else 1.0)


def update(elo_home, elo_away, home_margin, neutral: bool, playoff: bool, p: EloParams):
    """Return (home shift, home win probability). Works on scalars or arrays."""
    d = game_diff(elo_home, elo_away, neutral, playoff, p)
    prob = win_prob(d)
    result = np.where(home_margin > 0, 1.0, np.where(home_margin < 0, 0.0, 0.5))
    winner_d = np.where(home_margin >= 0, d, -d)
    shift = p.k * mov_mult(home_margin, winner_d) * (result - prob)
    return shift, prob


def regress(ratings: dict[str, float], p: EloParams) -> dict[str, float]:
    return {t: MEAN + (1 - p.regress) * (r - MEAN) for t, r in ratings.items()}


def played_games(schedules: pd.DataFrame) -> pd.DataFrame:
    g = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()].copy()
    g["neutral"] = g["location"] == "Neutral"
    g["playoff"] = g["game_type"].isin(PLAYOFF_TYPES)
    g["margin"] = g["home_score"] - g["away_score"]
    return g.reset_index(drop=True)


def run_history(schedules: pd.DataFrame, p: EloParams, until_game: str | None = None,
                start_season: int = 2002):
    """Replay real results in order.

    Returns (ratings before `until_game` (or at the end), per-game log DataFrame).
    """
    ratings = {t: MEAN for t in TEAMS}
    season = None
    log = []
    games = played_games(schedules)
    games = games[games["season"] >= start_season]
    for g in games.itertuples(index=False):
        if g.season != season:
            if season is not None:
                ratings = regress(ratings, p)
            season = g.season
        if g.game_id == until_game:
            return ratings, pd.DataFrame(log)
        h, a = g.home_team, g.away_team
        shift, prob = update(ratings[h], ratings[a], g.margin, g.neutral, g.playoff, p)
        log.append({"game_id": g.game_id, "season": g.season, "week": g.week, "game_type": g.game_type,
                    "home": h, "away": a, "home_elo": ratings[h], "away_elo": ratings[a],
                    "home_prob": float(prob), "margin": g.margin,
                    "home_elo_post": ratings[h] + float(shift), "away_elo_post": ratings[a] - float(shift)})
        ratings[h] += float(shift)
        ratings[a] -= float(shift)
    if until_game is not None:
        raise ValueError(f"game {until_game} not found in schedules")
    return ratings, pd.DataFrame(log)


def score(log: pd.DataFrame, seasons=(2010, 2019)) -> dict[str, float]:
    x = log[log["season"].between(*seasons)]
    y = np.where(x["margin"] > 0, 1.0, np.where(x["margin"] < 0, 0.0, 0.5))
    p = x["home_prob"].to_numpy()
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-9
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    decided = y != 0.5
    acc = float(np.mean((p[decided] > 0.5) == (y[decided] == 1)))
    return {"games": int(len(x)), "brier": brier, "log_loss": logloss, "accuracy": acc}


def calibrate(schedules: pd.DataFrame, verbose: bool = False):
    """Grid search on Brier score over 2010-2019 games (ratings warm up from 2002)."""
    grid = itertools.product([16, 20, 24], [40, 48, 55, 65], [1 / 3, 0.4, 0.5, 0.6, 0.7], [1.0, 1.2])
    results = []
    for k, hfa, reg, pm in grid:
        p = EloParams(k, hfa, reg, pm)
        _, log = run_history(schedules, p)
        s = score(log)
        results.append((s["brier"], p, s))
        if verbose:
            print(f"{p.label():<55} brier {s['brier']:.4f}")
    results.sort(key=lambda r: r[0])
    return results


def load_params(refresh: bool = False) -> EloParams:
    """Calibrated parameters (cached as JSON)."""
    path = data.cache_dir() / "elo_params.json"
    if path.exists() and not refresh:
        return EloParams(**json.loads(path.read_text())["params"])
    sched = data.load_schedules()
    best = calibrate(sched)
    brier, p, s = best[0]
    _, default_log = run_history(sched, EloParams())
    home = default_log[default_log["season"].between(2010, 2019)]
    y = np.where(home["margin"] > 0, 1.0, np.where(home["margin"] < 0, 0.0, 0.5))
    base = {"accuracy": float((y[y != 0.5] == 1).mean()), "brier": float(np.mean((y.mean() - y) ** 2))}
    path.write_text(json.dumps({"params": asdict(p), "score": s, "fivethirtyeight_default": score(default_log),
                                "home_team_baseline": base}, indent=2))
    return p


def calibration_report() -> str:
    path = data.cache_dir() / "elo_params.json"
    if not path.exists():
        load_params()
    info = json.loads(path.read_text())
    p = EloParams(**info["params"])
    s, d = info["score"], info["fivethirtyeight_default"]
    return (f"Elo calibrated on {s['games']} games, 2010-2019: {p.label()}\n"
            f"  Brier {s['brier']:.4f}, log loss {s['log_loss']:.4f}, picks the winner {s['accuracy']:.1%}\n"
            f"  538 defaults (K=20, HFA=65, 1/3, x1.2): Brier {d['brier']:.4f}, accuracy {d['accuracy']:.1%}\n"
            f"  always pick the home team: accuracy {info['home_team_baseline']['accuracy']:.1%}, "
            f"Brier {info['home_team_baseline']['brier']:.4f}")


def team_series(log: pd.DataFrame, team: str, seasons) -> pd.DataFrame:
    """Real rating after each game of `team` (plus each season's regressed start)."""
    rows = []
    for season in seasons:
        x = log[(log["season"] == season) & ((log["home"] == team) | (log["away"] == team))]
        if x.empty:
            continue
        first = x.iloc[0]
        start = first["home_elo"] if first["home"] == team else first["away_elo"]
        rows.append({"season": season, "week": 0, "elo": start})
        for r in x.itertuples(index=False):
            week = {"WC": 19, "DIV": 20, "CON": 21, "SB": 22}.get(r.game_type, r.week)
            rows.append({"season": season, "week": min(int(week), 22),
                         "elo": r.home_elo_post if r.home == team else r.away_elo_post})
    return pd.DataFrame(rows)


def elo_to_spread(diff: float) -> float:
    """538 rule of thumb: 25 Elo points ~ 1 point of spread."""
    return diff / 25.0
