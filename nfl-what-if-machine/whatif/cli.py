"""Command line for the NFL What If Machine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data
from .rewrite import OUTCOMES, Edit
from .teams import resolve_team

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 110)


def _pairs(text: str | None) -> dict[str, int] | None:
    """'SEA=24,NE=28' -> {'SEA': 24, 'NE': 28}"""
    if not text:
        return None
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[resolve_team(k)] = int(v)
    return out


def _add_edit_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("edit (what happens on the play instead)")
    g.add_argument("--outcome", choices=list(OUTCOMES), help="new result of the play")
    g.add_argument("--yards", type=int, help="yards for gain / sack / punt / penalty / turnover spot")
    g.add_argument("--penalty-on", default="defense", choices=["offense", "defense"])
    g.add_argument("--automatic-first-down", action="store_true", help="penalty gives an automatic first down")
    g.add_argument("--clock-secs", type=float, help="seconds the new play takes off the clock")
    s = p.add_argument_group("state overrides (after the play; before it for run / pass / custom)")
    s.add_argument("--possession", help="team with the ball")
    s.add_argument("--down", type=int)
    s.add_argument("--ydstogo", type=int, help="distance")
    s.add_argument("--yardline", help="'NE 1', 'own 20', 'opp 35' or yards to the opponent's goal")
    s.add_argument("--score", help="e.g. SEA=24,NE=28")
    s.add_argument("--clock", help="e.g. 'Q4 0:20'")
    s.add_argument("--timeouts", help="e.g. SEA=1,NE=2")


def _edit_from_args(a) -> Edit:
    if not a.outcome:
        raise SystemExit("--outcome is required (see --help for the list)")
    return Edit(outcome=a.outcome, yards=a.yards, penalty_on=a.penalty_on,
                automatic_first_down=a.automatic_first_down, clock_secs=a.clock_secs,
                possession=a.possession, down=a.down, ydstogo=a.ydstogo, yardline=a.yardline,
                score=_pairs(a.score), clock=a.clock, timeouts=_pairs(a.timeouts))


def _parse_adjust(text: str):
    """'SEA:2016:+20:Lynch does not retire' (optional ':week=N', ':if=SEA', ':world=both')."""
    from .seasonsim import Adjustment
    parts = text.split(":")
    if len(parts) < 3:
        raise SystemExit(f"--adjust needs TEAM:SEASON:ELO[:note], got {text!r}")
    kw = {"week": 1, "world": "alternate", "if_winner": None}
    note = []
    for extra in parts[3:]:
        if extra.startswith("week="):
            kw["week"] = int(extra[5:])
        elif extra.startswith("if="):
            kw["if_winner"] = resolve_team(extra[3:])
        elif extra.startswith("world="):
            kw["world"] = extra[6:]
        else:
            note.append(extra)
    return Adjustment(resolve_team(parts[0]), int(parts[1]), float(parts[2]), ":".join(note), **kw)


def _resolve_game(a) -> str:
    if getattr(a, "game_id", None):
        return a.game_id
    games = data.find_games(data.load_schedules(), a.season, a.week, resolve_team(a.team) if a.team else None)
    if len(games) != 1:
        print(games[["game_id", "game_type", "gameday", "away_team", "away_score", "home_team", "home_score"]]
              .to_string(index=False))
        raise SystemExit(f"{len(games)} games match; pick one with --game-id")
    return games.iloc[0]["game_id"]


# ---- commands ------------------------------------------------------------------------------
def cmd_games(a) -> None:
    g = data.find_games(data.load_schedules(), a.season, a.week, resolve_team(a.team) if a.team else None,
                        a.game_type)
    print(g[["game_id", "game_type", "week", "gameday", "away_team", "away_score", "home_team", "home_score"]]
          .to_string(index=False))


def cmd_plays(a) -> None:
    game = data.load_game(_resolve_game(a))
    plays = data.find_plays(game, a.qtr, a.clock, a.search, a.min_wpa)
    if a.top:
        plays = plays.reindex(plays["wpa"].abs().sort_values(ascending=False).index).head(a.top)
    print(data.play_summary(plays).to_string(index=False))


def cmd_game(a) -> None:
    from . import charts
    from .rewrite import rewrite_game
    game_id = _resolve_game(a)
    rw = rewrite_game(game_id, a.play_id, _edit_from_args(a), n=a.sims, seed=a.seed, workers=a.workers)
    print(rw.summary())
    out = Path(a.out or f"output/{game_id}_{a.play_id}_{a.outcome}")
    out.mkdir(parents=True, exist_ok=True)
    team = resolve_team(a.chart_team) if a.chart_team else None
    charts.win_probability(rw, team=team, path=str(out / "win_probability.png"))
    print(f"\nSaved {out / 'win_probability.png'}")


def cmd_run(a) -> None:
    from .scenario import Scenario, load_scenario, run_scenario, save_outputs
    if a.scenario:
        scn = load_scenario(a.scenario)
    else:
        if a.play_id is None:
            raise SystemExit("give a scenario file, or --game-id/--season... with --play-id and --outcome")
        scn = Scenario(name="what-if", game_id=_resolve_game(a), play_id=a.play_id, edit=_edit_from_args(a))
    if a.game_sims:
        scn.game_sims = a.game_sims
    if a.season_sims:
        scn.season_sims = a.season_sims
    if a.teams:
        scn.teams = [resolve_team(t) for t in a.teams.split(",")]
    scn.adjustments += [_parse_adjust(x) for x in a.adjust or []]
    if a.no_narrative:
        scn.adjustments = []
    res = run_scenario(scn, workers=a.workers, progress=lambda m: print(f"... {m}", file=sys.stderr))
    print(res.text_report())
    stem = Path(a.scenario).stem if a.scenario else f"{scn.game_id}_{scn.play_id}_{scn.edit.outcome}"
    paths = save_outputs(res, a.out or f"output/{stem}")
    print("\nSaved:")
    for p in paths.values():
        print(f"  {p}")


def cmd_elo(a) -> None:
    from . import elo
    elo.load_params(refresh=a.recalibrate)
    print(elo.calibration_report())


def cmd_validate(a) -> None:
    from .validate import report, validate
    print(report(validate(a.season, a.states, a.sims)))


def cmd_download(a) -> None:
    data.load_schedules(refresh=a.refresh)
    for season in range(data.FIRST_SEASON, data.LAST_SEASON + 1):
        data.load_pbp_season(season, refresh=a.refresh)
        print(f"  play-by-play {season} cached")
    print(f"Cache: {data.cache_dir()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="whatif", description="NFL What If Machine (2010-2019)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def game_filters(sp, need_play=False):
        sp.add_argument("--game-id", help="nflverse game id, e.g. 2014_21_NE_SEA")
        sp.add_argument("--season", type=int)
        sp.add_argument("--week", type=int)
        sp.add_argument("--team")
        if need_play:
            sp.add_argument("--play-id", type=int, help="play id (see `whatif plays`)")

    sp = sub.add_parser("games", help="list games")
    sp.add_argument("--season", type=int, required=True)
    sp.add_argument("--week", type=int)
    sp.add_argument("--team")
    sp.add_argument("--game-type", help="REG, WC, DIV, CON, SB")
    sp.set_defaults(func=cmd_games)

    sp = sub.add_parser("plays", help="find plays in a game")
    game_filters(sp)
    sp.add_argument("--qtr", type=int)
    sp.add_argument("--clock", help="game clock in the quarter, e.g. 0:26 (matches within 30 s)")
    sp.add_argument("--search", help="text in the play description")
    sp.add_argument("--min-wpa", type=float, help="only plays that moved win probability at least this much")
    sp.add_argument("--top", type=int, help="the N biggest win-probability swings")
    sp.set_defaults(func=cmd_plays)

    sp = sub.add_parser("game", help="Part 1 only: rewrite a play and simulate the rest of the game")
    game_filters(sp, need_play=True)
    _add_edit_args(sp)
    sp.add_argument("--sims", type=int, default=10_000)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--chart-team", help="team whose win probability is charted (default: offense)")
    sp.add_argument("--workers", type=int)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_game)

    sp = sub.add_parser("run", help="Parts 1 and 2: rewrite the game, then simulate the next seasons")
    sp.add_argument("scenario", nargs="?", help="scenario TOML file (see scenarios/)")
    game_filters(sp, need_play=True)
    _add_edit_args(sp)
    sp.add_argument("--game-sims", type=int)
    sp.add_argument("--season-sims", type=int)
    sp.add_argument("--teams", help="teams to report, e.g. SEA,NE")
    sp.add_argument("--adjust", action="append",
                    help="narrative adjustment TEAM:SEASON:ELO:note[:week=N][:if=TEAM][:world=both]")
    sp.add_argument("--no-narrative", action="store_true", help="ignore the scenario's adjustments")
    sp.add_argument("--workers", type=int)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("elo", help="show (or redo) the Elo calibration")
    sp.add_argument("--recalibrate", action="store_true")
    sp.set_defaults(func=cmd_elo)

    sp = sub.add_parser("validate", help="check the game simulator against real outcomes")
    sp.add_argument("--season", type=int, default=2018)
    sp.add_argument("--states", type=int, default=400)
    sp.add_argument("--sims", type=int, default=500)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("download", help="download and cache all data")
    sp.add_argument("--refresh", action="store_true")
    sp.set_defaults(func=cmd_download)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ValueError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
