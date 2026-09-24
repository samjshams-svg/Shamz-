"""Check the game simulator against real outcomes (and nflfastR) on random real situations."""

from __future__ import annotations

import multiprocessing as mp
import os
import random

import numpy as np
import pandas as pd

from . import data
from .models import load_model
from .rewrite import _init, _simulate, state_from_row


def validate(season: int = 2018, n_states: int = 400, n_sims: int = 500, seed: int = 7) -> pd.DataFrame:
    pbp = data.load_pbp_season(season)
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["result"] != 0) & pbp["home_wp"].notna()]
    rows = pbp.sample(n=n_states * 2, random_state=seed)
    states, keep = [], []
    for _, r in rows.iterrows():
        s = state_from_row(r)
        if s is not None:
            states.append(s)
            keep.append(r)
        if len(states) == n_states:
            break
    model = load_model(season)
    jobs = [(s, n_sims, seed * 100_000 + i, False) for i, s in enumerate(states)]
    with mp.get_context("fork").Pool(os.cpu_count(), initializer=_init, initargs=(model,)) as pool:
        parts = pool.map(_simulate, jobs, chunksize=4)
    out = []
    for s, r, part in zip(states, keep, parts):
        home_wins = sum(1 for w, _, _ in part if w == s.home) + 0.5 * sum(1 for w, _, _ in part if w is None)
        out.append({"game_id": r["game_id"], "play_id": r["play_id"], "minutes": s.elapsed_minutes,
                    "model_wp": home_wins / len(part), "nflfastr_wp": r["home_wp"],
                    "home_won": float(r["result"] > 0)})
    return pd.DataFrame(out)


def report(df: pd.DataFrame) -> str:
    def brier(col):
        return float(((df[col] - df["home_won"]) ** 2).mean())

    lines = [f"{len(df)} random real situations",
             f"  Brier score, this model: {brier('model_wp'):.4f}",
             f"  Brier score, nflfastR  : {brier('nflfastr_wp'):.4f}",
             f"  Brier score, coin flip : 0.2500",
             f"  mean |model - nflfastR|: {(df['model_wp'] - df['nflfastr_wp']).abs().mean():.3f}",
             "", "  calibration (model WP bucket -> share of home wins):"]
    bins = pd.cut(df["model_wp"], [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    for b, g in df.groupby(bins, observed=True):
        lines.append(f"    {str(b):<14} n={len(g):>4}  predicted {g['model_wp'].mean():.2f}  actual {g['home_won'].mean():.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    print(report(validate()))
