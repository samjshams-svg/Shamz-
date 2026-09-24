"""Scenario files (TOML) and the full pipeline: rewrite the game, then replay history."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import data, elo as E
from .rewrite import Edit, GameRewrite, rewrite_game
from .seasonsim import Adjustment, Change, HistorySim, WorldResult, actual_outcomes
from .teams import TEAM_INDEX, resolve_team

METRICS = [("wins", "Wins"), ("playoffs", "Playoffs"), ("division", "Division"),
           ("conference", "Conf. title"), ("champion", "Super Bowl")]


@dataclass
class Scenario:
    name: str
    game_id: str
    play_id: int
    edit: Edit
    game_sims: int = 10_000
    season_sims: int = 10_000
    seasons_ahead: int = 2
    seed: int = 49
    teams: list[str] = field(default_factory=list)
    adjustments: list[Adjustment] = field(default_factory=list)


def load_scenario(path: str | Path) -> Scenario:
    raw = tomllib.loads(Path(path).read_text())
    return scenario_from_dict(raw, default_name=Path(path).stem)


def scenario_from_dict(raw: dict, default_name: str = "scenario") -> Scenario:
    play = raw.get("play", {})
    game_id, play_id = resolve_play(play)
    ed = dict(raw.get("edit", {}))
    if "outcome" not in ed:
        raise ValueError("[edit] needs an outcome")
    edit = Edit(**ed)
    sim = raw.get("simulation", {})
    adjustments = [Adjustment(team=resolve_team(a["team"]), season=int(a["season"]), elo=float(a["elo"]),
                              note=a.get("note", ""), week=int(a.get("week", 1)),
                              world=a.get("world", "alternate"),
                              if_winner=resolve_team(a["if_winner"]) if a.get("if_winner") else None)
                   for a in raw.get("adjustment", [])]
    return Scenario(
        name=raw.get("name", default_name), game_id=game_id, play_id=play_id, edit=edit,
        game_sims=int(sim.get("game_sims", 10_000)), season_sims=int(sim.get("season_sims", 10_000)),
        seasons_ahead=int(sim.get("seasons_ahead", 2)), seed=int(sim.get("seed", 49)),
        teams=[resolve_team(t) for t in sim.get("teams", [])], adjustments=adjustments,
    )


def resolve_play(play: dict) -> tuple[str, int]:
    """[play] needs game_id + play_id, or filters that match exactly one play."""
    game_id = play.get("game_id")
    if not game_id:
        sched = data.load_schedules()
        team = resolve_team(play["team"]) if play.get("team") else None
        games = data.find_games(sched, play.get("season"), play.get("week"), team)
        if play.get("opponent"):
            opp = resolve_team(play["opponent"])
            games = games[(games["home_team"] == opp) | (games["away_team"] == opp)]
        if len(games) != 1:
            raise ValueError(f"[play] matches {len(games)} games; add season/week/team/opponent or game_id: "
                             + ", ".join(games["game_id"].head(10)))
        game_id = games.iloc[0]["game_id"]
    if play.get("play_id"):
        return game_id, int(play["play_id"])
    plays = data.find_plays(data.load_game(game_id), play.get("qtr"), play.get("clock"), play.get("search"),
                            play.get("min_abs_wpa"))
    if len(plays) != 1:
        rows = data.play_summary(plays).head(8).to_string(index=False)
        raise ValueError(f"[play] matches {len(plays)} plays in {game_id}; narrow it or give play_id:\n{rows}")
    return game_id, int(plays.iloc[0]["play_id"])


@dataclass
class ScenarioResult:
    scenario: Scenario
    game: GameRewrite
    alternate: WorldResult
    baseline: WorldResult
    actual: dict[int, pd.DataFrame]
    elo_params: E.EloParams
    elo_log: pd.DataFrame
    start_ratings: dict[str, float]
    teams: list[str]

    def odds_table(self, teams: list[str] | None = None) -> pd.DataFrame:
        teams = teams or self.teams
        rows = []
        for alt, base in zip(self.alternate.seasons, self.baseline.seasons):
            act = self.actual[alt.season]
            for t in teams:
                i = TEAM_INDEX[t]
                row = {"season": alt.season, "team": t}
                for key, _ in METRICS:
                    a = getattr(alt, key)[:, i].mean()
                    b = getattr(base, key)[:, i].mean()
                    row[f"actual_{key}"] = float(act.loc[t, key])
                    row[f"base_{key}"] = float(b)
                    row[f"alt_{key}"] = float(a)
                rows.append(row)
        return pd.DataFrame(rows)

    def movers(self, top: int = 5) -> pd.DataFrame:
        """Teams whose playoff odds moved most (outside the report teams)."""
        rows = []
        for alt, base in zip(self.alternate.seasons[1:], self.baseline.seasons[1:]):
            d = alt.playoffs.mean(0) - base.playoffs.mean(0)
            for i in np.argsort(-np.abs(d))[: top + len(self.teams)]:
                team = list(TEAM_INDEX)[i]
                if team in self.teams:
                    continue
                rows.append({"season": alt.season, "team": team, "base_playoffs": base.playoffs[:, i].mean(),
                             "alt_playoffs": alt.playoffs[:, i].mean(), "change": d[i]})
        return pd.DataFrame(rows).groupby("season").head(top).reset_index(drop=True) if rows else pd.DataFrame()

    def text_report(self) -> str:
        lines = [f"# {self.scenario.name}", "", "## Part 1: the game", "", self.game.summary(), "",
                 "## Part 2: the next seasons", "",
                 f"{self.scenario.season_sims:,} simulated histories per world. Elo: {self.elo_params.label()}.",
                 f"Ratings just before the changed game: " +
                 ", ".join(f"{t} {self.start_ratings[t]:.0f}" for t in self.teams), ""]
        if self.scenario.adjustments:
            lines.append("Narrative adjustments:")
            for a in self.scenario.adjustments:
                cond = f", only if {a.if_winner} won the changed game" if a.if_winner else ""
                lines.append(f"  {a.team} {a.elo:+g} Elo from {a.season} week {a.week} ({a.world}{cond}): {a.note}")
            lines.append("")
        lines.append(format_odds(self.odds_table()))
        mv = self.movers()
        if not mv.empty:
            lines += ["", "Other teams whose playoff odds moved most:"]
            for r in mv.itertuples(index=False):
                lines.append(f"  {r.season} {r.team:<4} {r.base_playoffs:6.1%} -> {r.alt_playoffs:6.1%} ({r.change:+.1%})")
        return "\n".join(lines)


def format_odds(table: pd.DataFrame) -> str:
    head = f"{'season':<7}{'team':<6}{'':<9}" + "".join(f"{label:>12}" for _, label in METRICS)
    lines = [head, "-" * len(head)]
    for r in table.to_dict("records"):
        for world, label in (("actual", "actual"), ("base", "no change"), ("alt", "with edit")):
            cells = []
            for key, _ in METRICS:
                v = r[f"{world}_{key}"]
                if key == "wins":
                    cells.append(f"{v:12.1f}")
                elif world == "actual":
                    cells.append(f"{'yes' if v >= 0.5 else '-':>12}")
                else:
                    cells.append(f"{v:12.1%}")
            first = f"{r['season']:<7}{r['team']:<6}" if world == "actual" else " " * 13
            lines.append(f"{first}{label:<9}" + "".join(cells))
    return "\n".join(lines)


def run_scenario(scn: Scenario, workers: int | None = None,
                 progress: Callable[[str], None] | None = None) -> ScenarioResult:
    say = progress or (lambda msg: None)
    say("Rewriting the game")
    game = rewrite_game(scn.game_id, scn.play_id, scn.edit, n=scn.game_sims, seed=scn.seed, workers=workers)
    sched = data.load_schedules()
    params = E.load_params()
    say("Replaying real history up to the changed game")
    start, _ = E.run_history(sched, params, until_game=scn.game_id)
    _, log = E.run_history(sched, params)
    row = sched[sched["game_id"] == scn.game_id].iloc[0]
    if row["home_team"] != game.home:
        raise ValueError("schedule and play-by-play disagree on the home team")
    rng = np.random.default_rng(scn.seed)
    if len(game.alt.margins) == scn.season_sims:
        alt_margins = game.alt.margins.astype(float)   # one simulated history per simulated game
    else:
        alt_margins = rng.choice(game.alt.margins, size=scn.season_sims).astype(float)
    real_margin = float(row["home_score"] - row["away_score"])
    season = int(row["season"])
    sim = HistorySim(sched, params, scn.season_sims, seed=scn.seed)
    say(f"Simulating {scn.season_sims:,} alternate histories")
    alt = sim.run(start, Change(scn.game_id, season, row["game_type"], game.home, game.away, alt_margins),
                  "alternate", scn.seasons_ahead, scn.adjustments)
    say(f"Simulating {scn.season_sims:,} histories without the change")
    base = sim.run(start, Change(scn.game_id, season, row["game_type"], game.home, game.away,
                                 np.full(scn.season_sims, real_margin)),
                   "baseline", scn.seasons_ahead, scn.adjustments)
    actual = {s: actual_outcomes(sched, s) for s in range(season, season + scn.seasons_ahead + 1)}
    teams = scn.teams or [game.pre.offense, game.pre.defense]
    return ScenarioResult(scn, game, alt, base, actual, params, log, start, teams)


def save_outputs(res: ScenarioResult, out_dir: str | Path) -> dict[str, Path]:
    from . import charts
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": out / "report.md",
        "odds_csv": out / "odds.csv",
        "win_probability": out / "win_probability.png",
        "elo": out / "elo_trajectory.png",
        "odds_chart": out / "season_odds.png",
        "summary": out / "summary.json",
    }
    paths["report"].write_text(res.text_report().replace("\n", "  \n") + "\n")
    res.odds_table().to_csv(paths["odds_csv"], index=False)
    charts.win_probability(res.game, team=res.teams[0], path=str(paths["win_probability"]))
    charts.elo_trajectory(res, path=str(paths["elo"]))
    charts.odds_table(res, path=str(paths["odds_chart"]))
    g = res.game
    summary = {
        "scenario": res.scenario.name, "game_id": g.game_id, "play_id": g.play_id, "edit": g.edit.describe(),
        "win_probability": {t: g.alt.prob(t) for t in (g.home, g.away)} | {"tie": g.alt.tie_prob},
        "nflfastr_after_real_play": {g.home: g.nflfastr_after, g.away: 1 - g.nflfastr_after},
        "odds": res.odds_table().to_dict("records"),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, default=float))
    return paths
