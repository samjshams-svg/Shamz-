"""Command line: change one play of Super Bowl XLIX and re-simulate the rest."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from . import data, game, plot
from .model import LeagueModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    outcomes = "\n".join(f"  {k:<18} {v}" for k, v in game.OUTCOMES.items())
    p = argparse.ArgumentParser(
        prog="python -m sb49",
        description="Change the result of one Super Bowl XLIX play and estimate who wins from there.",
        epilog=f"outcomes:\n{outcomes}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="list the game's plays with their play ids and exit")
    p.add_argument("--play-id", type=int, default=4205,
                   help="play to change (default 4205: 2nd & goal at the NE 1, the Butler interception)")
    p.add_argument("--outcome", default="touchdown", choices=sorted(game.OUTCOMES), help="new result of the play")
    p.add_argument("--yards", type=int, help="yards for gain / punt / turnover spot")
    p.add_argument("--clock-secs", type=float,
                   help="seconds the new play takes off the clock (default: 5-8 for clock-stopping results, "
                        "the real play's run-off otherwise)")
    p.add_argument("--team", help="team whose win probability is plotted (default: offense on the changed play)")
    p.add_argument("--sims", type=int, default=20_000, help="simulations after the changed play (default 20000)")
    p.add_argument("--curve-sims", type=int, default=1_000,
                   help="simulations per real play for the simulated win-probability line (default 1000)")
    p.add_argument("--no-curve", action="store_true", help="skip the per-play simulated line (much faster)")
    p.add_argument("--seed", type=int, default=49)
    p.add_argument("--workers", type=int, help="processes to use (default: all cores)")
    p.add_argument("--out", help="output PNG path (default: plots/sb49_<play>_<outcome>.png)")
    return p.parse_args(argv)


def list_plays(g) -> None:
    for _, r in g.iterrows():
        if r["play_type"] in (None, "no_play") and "Timeout" not in str(r["desc"]):
            continue
        pid = int(r["play_id"])
        qtr = "" if np.isnan(r["qtr"]) else f"Q{int(r['qtr'])}"
        pos = r["posteam"] or ""
        print(f"{pid:>5}  {qtr:<3} {pos:<4} {str(r['desc'])[:110]}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    pbp = data.load_season()
    g = data.load_game(pbp)
    if args.list:
        list_plays(g)
        return

    model = LeagueModel.from_pbp(pbp, exclude_game=data.GAME_ID)
    idx = game.play_row(g, args.play_id)
    row = g.loc[idx]
    pre = game.state_from_row(row)
    if pre is None:
        raise SystemExit(f"play {args.play_id} is not a snap (timeout, penalty-only or period marker)")

    home, away = row["home_team"], row["away_team"]
    team = args.team or pre.offense
    opp = away if team == home else home

    ov = game.Override(args.outcome, args.yards, args.clock_secs)
    post, winner = game.apply_override(pre, ov, game.real_duration(g, idx), model)

    print(f"Calibrated on {len(pbp['game_id'].unique()) - 1} games of {data.SEASON} play-by-play\n")
    print(f"Play {args.play_id}: {row['desc']}")
    print(f"  before : {pre.describe()}")
    print(f"  changed: {args.outcome}" + (f" ({args.yards:+d} yds)" if args.yards is not None else ""))
    if winner:
        print(f"  after  : game over, {winner} wins")
        cf = game.MCResult({winner: args.sims}, args.sims,
                           np.full(args.sims, post.score[team] - post.score[opp]), (team, opp))
    else:
        print(f"  after  : {post.describe()}")
        cf = game.monte_carlo(post, model, args.sims, seed=args.seed, workers=args.workers)

    real_pre = row["home_wp"] if team == home else 1 - row["home_wp"]
    real_post = row["home_wp_post"] if team == home else 1 - row["home_wp_post"]
    print(f"\nMonte Carlo win probability after the changed play ({cf.n:,} sims):")
    for t in (team, opp):
        lo, hi = cf.ci95(t)
        print(f"  {t}: {cf.prob(t):6.1%}   (95% CI {lo:.1%} - {hi:.1%})")
    print(f"nflfastR {team} win probability: {real_pre:.1%} before the snap, {real_post:.1%} after what really happened")

    margins = cf.margins if cf.teams[0] == team else -cf.margins

    # Simulated line: Monte Carlo from every real pre-snap state in the game.
    snaps = [(i, s) for i, s in ((i, game.state_from_row(r)) for i, r in g.iterrows()) if s is not None]
    if args.no_curve:
        sim_x, sim_wp = np.array([]), np.array([])
    else:
        print(f"\nSimulating {len(snaps)} real game states x {args.curve_sims:,} sims for the comparison line...")
        res = game.monte_carlo_many([s for _, s in snaps], model, args.curve_sims, seed=args.seed, workers=args.workers)
        sim_x = plot.minutes_elapsed(g.loc[[i for i, _ in snaps], "game_seconds_remaining"])
        sim_wp = np.array([r.prob(team) for r in res])
        real_at = g.loc[[i for i, _ in snaps], "home_wp"].to_numpy()
        real_at = real_at if team == home else 1 - real_at
        err = np.abs(sim_wp - real_at)
        print(f"  mean |Monte Carlo - nflfastR| = {np.nanmean(err):.1%}, max = {np.nanmax(err):.1%}")

    real = g[g["home_wp"].notna()]
    real_wp = real["home_wp"].to_numpy() if team == home else 1 - real["home_wp"].to_numpy()
    actual_margin = _final_margin(g, team)

    label = f"Q{int(row['qtr'])}, {pre.offense} {pre.down}&{pre.ydstogo} at {pre.spot()}. Real: " + _short(row["desc"])
    new_label = "Changed to " + args.outcome.replace("_", " ") + (f" ({args.yards:+d})" if args.yards is not None else "")
    out = Path(args.out or f"plots/sb49_{args.play_id}_{args.outcome}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.render(plot.ChartData(
        team=team, opponent=opp,
        real_x=plot.minutes_elapsed(real["game_seconds_remaining"]), real_wp=real_wp,
        sim_x=sim_x, sim_wp=sim_wp,
        play_x=float(plot.minutes_elapsed([post.game_secs])[0]) if post.half <= 2 else 60.0,
        play_label=label, new_label=new_label,
        actual_post_wp=float(real_post), cf_wp=cf.prob(team), cf_ci=cf.ci95(team),
        cf_margins=margins, actual_margin=actual_margin, n_sims=cf.n,
    ), str(out))
    print(f"\nSaved {out}  ({time.time() - t0:.0f}s)")


def _final_margin(g, team: str) -> int:
    last = g[g["posteam_score"].notna()].iloc[-1]
    # posteam_score is pre-play; the final row carries the final score.
    pts = {last["posteam"]: int(last["posteam_score"]), last["defteam"]: int(last["defteam_score"])}
    opp = next(t for t in pts if t != team)
    return pts[team] - pts[opp]


def _short(desc: str, n: int = 110) -> str:
    desc = re.sub(r"^\([\d:]*\)\s*(\([^)]*\)\s*)*", "", str(desc))
    return desc if len(desc) <= n else desc[: n - 1] + "…"
