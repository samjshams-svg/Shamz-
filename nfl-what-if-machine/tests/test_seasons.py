import math
import os
import random

import numpy as np
import pandas as pd
import pytest

from whatif import elo as E
from whatif.seasonsim import Adjustment, Change, HistorySim, Standings, _SeedContext, actual_outcomes, real_seeds
from whatif.teams import TEAM_INDEX, TEAMS


def test_elo_update_matches_538_formula():
    p = E.EloParams(k=20, hfa=65, regress=1 / 3, playoff_mult=1.2)
    shift, prob = E.update(1500.0, 1500.0, 7, neutral=True, playoff=False, p=p)
    assert prob == pytest.approx(0.5)
    assert shift == pytest.approx(20 * math.log(8) * 0.5)
    # Home field counts unless the site is neutral.
    _, prob_home = E.update(1500.0, 1500.0, 7, neutral=False, playoff=False, p=p)
    assert prob_home == pytest.approx(1 / (1 + 10 ** (-65 / 400)))
    # A big favorite winning big moves less than an underdog winning big.
    fav, _ = E.update(1700.0, 1400.0, 21, neutral=True, playoff=False, p=p)
    dog, _ = E.update(1400.0, 1700.0, 21, neutral=True, playoff=False, p=p)
    assert 0 < fav < dog


def test_regression_to_mean():
    p = E.EloParams(regress=0.5)
    r = E.regress({"A": 1705.0, "B": 1305.0}, p)
    assert r == {"A": 1605.0, "B": 1405.0}


def _standings(results):
    """results: list of (home, away, margin) with team codes."""
    games = pd.DataFrame({"h": [TEAM_INDEX[h] for h, _, _ in results], "a": [TEAM_INDEX[a] for _, a, _ in results]})
    st = Standings(1, games)
    for h, a, m in results:
        st.add(TEAM_INDEX[h], TEAM_INDEX[a], np.array([m]))
    return st


def _ctx(st):
    g = np.maximum(st.games, 1)
    pct = st.W2 / (2 * g)
    dp = st.DW2 / (2 * np.maximum(st.div_games, 1))
    cp = st.CW2 / (2 * np.maximum(st.conf_games, 1))
    zeros = [0.0] * len(TEAMS)
    return _SeedContext(st.W2[0].tolist(), dp[0].tolist(), cp[0].tolist(), zeros, zeros, st.HW2[0].tolist(),
                        st.G.tolist(), random.Random(0)), pct


def test_head_to_head_breaks_division_tie():
    # SEA and SF both 2-2; SEA swept SF.
    st = _standings([("SEA", "SF", 3), ("SF", "SEA", -2), ("SF", "ARI", 10), ("ARI", "SF", -3),
                     ("ARI", "SEA", 7), ("SEA", "ARI", -3)])
    ctx, _ = _ctx(st)
    assert st.W2[0][TEAM_INDEX["SEA"]] == st.W2[0][TEAM_INDEX["SF"]] == 4
    order = ctx.order([TEAM_INDEX[t] for t in ("SF", "SEA")], division=True)
    assert TEAMS[order[0]] == "SEA"


def test_wildcard_three_way_needs_a_sweep():
    # DEN, NE and CIN all 2-2. DEN beat both others; NE and CIN never met -> DEN by sweep.
    st = _standings([("DEN", "NE", 3), ("DEN", "CIN", 3), ("KC", "DEN", 3), ("LV", "DEN", 3),
                     ("NE", "NYJ", 3), ("NE", "MIA", 3), ("BUF", "NE", 3),
                     ("CIN", "PIT", 3), ("CIN", "BAL", 3), ("CLE", "CIN", 3)])
    ctx, _ = _ctx(st)
    group = [TEAM_INDEX[t] for t in ("NE", "CIN", "DEN")]
    assert all(st.W2[0][g] == 4 for g in group)
    assert TEAMS[ctx._best(group, division=False)] == "DEN"


needs_data = pytest.mark.skipif(os.environ.get("WHATIF_OFFLINE") == "1", reason="needs nflverse data")


@needs_data
@pytest.mark.parametrize("season", range(2010, 2022))
def test_seeding_reproduces_real_wild_card_round(season):
    from whatif import data
    from whatif.seasonsim import playoff_teams
    sched = data.load_schedules()
    afc, nfc = real_seeds(sched, season)
    pairs = [(2, 5), (3, 4)] if playoff_teams(season) == 6 else [(1, 6), (2, 5), (3, 4)]
    mine = {frozenset((s[a], s[b])) for s in (afc, nfc) for a, b in pairs}
    wc = sched[(sched["season"] == season) & (sched["game_type"] == "WC")]
    assert mine == {frozenset((r.home_team, r.away_team)) for r in wc.itertuples()}


@needs_data
def test_actual_outcomes_2015():
    from whatif import data
    a = actual_outcomes(data.load_schedules(), 2015)
    assert a.loc["DEN", "champion"] and a.loc["CAR", "conference"] and a.loc["NE", "division"]
    assert a.loc["SEA", "wins"] == 10 and a["champion"].sum() == 1 and a["playoffs"].sum() == 12


@needs_data
def test_history_sim_sb49():
    from whatif import data
    sched = data.load_schedules()
    p = E.load_params()
    start, _ = E.run_history(sched, p, until_game="2014_21_NE_SEA")
    sim = HistorySim(sched, p, n=500, seed=2)
    n = 500
    sea_wins = np.where(np.arange(n) < 400, 3.0, -4.0)   # SEA wins 80% of alternate histories
    adj = [Adjustment("SEA", 2015, 1000, "huge bump, only when SEA won", if_winner="SEA"),
           Adjustment("NE", 2015, 1000, "only in the other world", world="baseline")]
    alt = sim.run(start, Change("2014_21_NE_SEA", 2014, "SB", "SEA", "NE", sea_wins), "alternate", 2, adj)
    base = sim.run(start, Change("2014_21_NE_SEA", 2014, "SB", "SEA", "NE", np.full(n, -4.0)), "baseline", 2, adj)
    s14 = alt.seasons[0]
    assert s14.champion[:, TEAM_INDEX["SEA"]].mean() == pytest.approx(0.8)
    assert base.seasons[0].champion[:, TEAM_INDEX["NE"]].all()
    # Real 2014 standings are fixed in every history.
    assert (s14.wins[:, TEAM_INDEX["SEA"]] == 12).all()
    # +1000 Elo only in histories where SEA won -> those SEA teams win (nearly) everything in 2015.
    w15 = alt.seasons[1].wins[:, TEAM_INDEX["SEA"]]
    assert w15[:400].mean() > 15 and w15[400:].mean() < 13
    # The 'baseline'-only adjustment never reaches the alternate world.
    assert alt.seasons[1].wins[:, TEAM_INDEX["NE"]].mean() < 14
    for so in alt.seasons + base.seasons:
        assert so.champion.sum(1).tolist() == [1] * n
        assert (so.playoffs.sum(1) == 12).all()
        assert (so.division.sum(1) == 8).all()
